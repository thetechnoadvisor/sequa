from __future__ import annotations

import os
import time
from typing import Any, Callable

from sequa.llm.adapters.base import CanonicalRequest, CanonicalResponse, ProviderAdapter
from sequa.matcher import hash_request
from sequa.models import Cassette
from sequa import storage
from sequa.utils import serialize_type_or_pydantic


class SequaError(Exception):
    """Base exception for Sequa."""


class CassetteNotFoundError(SequaError):
    """Exception raised when a cassette is missing in replay mode."""


def to_serializable(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, type) or callable(val):
        return serialize_type_or_pydantic(val)
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(v) for v in val]
    if isinstance(val, tuple):
        return tuple(to_serializable(v) for v in val)
    if isinstance(val, set):
        return {to_serializable(v) for v in val}
    
    # Try pydantic / dict methods
    if hasattr(val, "model_dump"):
        try:
            return to_serializable(val.model_dump())
        except Exception:
            pass
    if hasattr(val, "dict"):
        try:
            return to_serializable(val.dict())
        except Exception:
            pass
    if hasattr(val, "to_dict"):
        try:
            return to_serializable(val.to_dict())
        except Exception:
            pass
            
    # Try custom object __dict__
    if hasattr(val, "__dict__"):
        return {
            "__class_info__": f"{val.__class__.__module__}.{val.__class__.__name__}",
            "data": {k: to_serializable(v) for k, v in val.__dict__.items()}
        }
        
    return str(val)


def from_serializable(val: Any) -> Any:
    if isinstance(val, dict):
        if "__class_info__" in val:
            class_path = val["__class_info__"]
            data = from_serializable(val["data"])
            import importlib
            try:
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                
                if hasattr(cls, "model_validate"):
                    return cls.model_validate(data)
                elif hasattr(cls, "parse_obj"):
                    return cls.parse_obj(data)
                elif hasattr(cls, "from_dict"):
                    return cls.from_dict(data)
                
                obj = cls.__new__(cls)
                obj.__dict__.update(data)
                return obj
            except Exception:
                return data
        else:
            return {k: from_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [from_serializable(v) for v in val]
    if isinstance(val, tuple):
        return tuple(from_serializable(v) for v in val)
    if isinstance(val, set):
        return {from_serializable(v) for v in val}
    return val


def serialize_chunk(chunk: Any) -> Any:
    if isinstance(chunk, (str, int, float, bool, type(None))):
        return chunk
        
    class_info = f"{chunk.__class__.__module__}.{chunk.__class__.__name__}"
    
    if hasattr(chunk, "model_dump"):
        try:
            return {
                "__type__": "pydantic",
                "class": class_info,
                "data": to_serializable(chunk.model_dump())
            }
        except Exception:
            pass

    if hasattr(chunk, "dict"):
        try:
            return {
                "__type__": "pydantic",
                "class": class_info,
                "data": to_serializable(chunk.dict())
            }
        except Exception:
            pass

    if hasattr(chunk, "to_dict"):
        try:
            return {
                "__type__": "dict_method",
                "class": class_info,
                "data": to_serializable(chunk.to_dict())
            }
        except Exception:
            pass

    if hasattr(chunk, "__dict__"):
        try:
            return {
                "__type__": "fallback",
                "class": class_info,
                "data": to_serializable(chunk.__dict__)
            }
        except Exception:
            pass

    return to_serializable(chunk)


def deserialize_chunk(serialized: Any) -> Any:
    if not isinstance(serialized, dict) or "__type__" not in serialized:
        return from_serializable(serialized)

    type_ = serialized.get("__type__")
    class_path = serialized.get("class")
    data = from_serializable(serialized.get("data"))
    if not class_path:
        return data

    import importlib
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, ValueError, AttributeError):
        return data

    if type_ == "pydantic":
        try:
            if hasattr(cls, "model_validate"):
                return cls.model_validate(data)
            elif hasattr(cls, "parse_obj"):
                return cls.parse_obj(data)
            return cls(**data)
        except Exception:
            return data
    elif type_ == "dict_method":
        try:
            if hasattr(cls, "from_dict"):
                return cls.from_dict(data)
            return cls(**data)
        except Exception:
            return data
    elif type_ == "fallback":
        try:
            obj = cls.__new__(cls)
            obj.__dict__.update(data)
            return obj
        except Exception:
            return data

    return data


def extract_text_from_chunk(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "content"):
        return str(chunk.content)
    if isinstance(chunk, dict):
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta:
                return str(delta["content"])
        if "content" in chunk:
            return str(chunk["content"])
    else:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and hasattr(delta, "content") and delta.content is not None:
                return str(delta.content)
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and chunk.delta.text is not None:
            return str(chunk.delta.text)
    return ""


class ReplayStream:
    def __init__(self, gen_obj: Any) -> None:
        self._gen = gen_obj

    def __iter__(self) -> ReplayStream:
        return self

    def __next__(self) -> Any:
        return next(self._gen)

    def close(self) -> None:
        if hasattr(self._gen, "close"):
            self._gen.close()

    def __enter__(self) -> ReplayStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ReplayAsyncStream:
    def __init__(self, async_gen_obj: Any) -> None:
        self._gen = async_gen_obj

    def __aiter__(self) -> ReplayAsyncStream:
        return self

    async def __anext__(self) -> Any:
        return await self._gen.__anext__()

    async def close(self) -> None:
        if hasattr(self._gen, "aclose"):
            await self._gen.aclose()
        elif hasattr(self._gen, "close"):
            self._gen.close()

    async def __aenter__(self) -> ReplayAsyncStream:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class RecorderEngine:
    def __init__(
        self,
        path: str,
        mode: str = "auto",
        ignore_fields: list[str] | None = None,
        normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        mask_pii: bool = False,
        guardrails: list[str] | dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.mode = mode.lower()
        self.ignore_fields = ignore_fields or []
        self.normalizer = normalizer
        self.mask_pii = mask_pii
        self.guardrails = guardrails

        from sequa.guardrails import NeMoGuardrailsEngine
        self.guardrails_engine = NeMoGuardrailsEngine(guardrails)

        if self.mode not in ("replay", "record", "auto", "live"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be replay, record, auto, or live.")

    def get_cassette_path(self, req_hash: str, provider: str = "") -> str:
        """Resolve the path to the cassette file based on request hash, provider, and configured path."""
        if self.path.lower().endswith(".json"):
            return self.path

        provider_dir = provider.strip().lower() if provider else "default"
        return os.path.join(self.path, provider_dir, f"{req_hash}.json")

    def find_cassette_path(self, req_hash: str, provider: str = "") -> str:
        """Find existing cassette path or return default destination for saving."""
        if self.path.lower().endswith(".json"):
            return self.path

        # 1. Check primary provider path
        if provider:
            provider_dir = provider.strip().lower()
            primary_path = os.path.join(self.path, provider_dir, f"{req_hash}.json")
            if storage.exists(primary_path):
                return primary_path

        # 2. Check legacy path directly under self.path
        legacy_path = os.path.join(self.path, f"{req_hash}.json")
        if storage.exists(legacy_path):
            return legacy_path

        # 3. Check any subfolder under self.path
        target_file = f"{req_hash}.json"
        if os.path.isdir(self.path):
            for root, _, files in os.walk(self.path):
                if target_file in files:
                    return os.path.join(root, target_file)

        # 4. Default saving path
        return self.get_cassette_path(req_hash, provider)


    def mask_pii_and_si(self, text: str) -> str:
        """Mask emails, phone numbers, and common PII/SI in a string."""
        if not isinstance(text, str):
            return text

        import re

        # 1. Mask Emails
        email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        text = re.sub(email_pattern, "[EMAIL]", text)

        # 2. Mask Phone Numbers
        phone_pattern = r"(?:\+?\b(?:\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b|\+?\b\d{2,4}[-.\s]?\d{5}[-.\s]?\d{5}\b|\+?\b\d{5}[-.\s]?\d{5}\b)"
        text = re.sub(phone_pattern, "[PHONE]", text)

        # 3. Mask Credit Cards (13 to 16 digits)
        credit_card_pattern = r"\b(?:\d[ -]*?){13,16}\b"
        text = re.sub(credit_card_pattern, "[CREDIT_CARD]", text)

        # 4. Mask SSNs
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        text = re.sub(ssn_pattern, "[SSN]", text)

        # 5. Mask IP Addresses
        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        text = re.sub(ip_pattern, "[IP_ADDRESS]", text)

        # 6. Mask API Keys / Secrets
        api_key_pattern = r"\b(?:sk-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9_\-]{35}|sk-ant-[a-zA-Z0-9\-]{20,})\b"
        text = re.sub(api_key_pattern, "[API_KEY]", text)

        # Generic bearer tokens
        bearer_pattern = r"\bBearer\s+[a-zA-Z0-9\-\._~\+\/]+=*"
        text = re.sub(bearer_pattern, "Bearer [TOKEN]", text)

        return text

    def mask_value(self, val: Any) -> Any:
        """Recursively mask strings, lists, dictionaries, tuples, sets, and message objects."""
        if isinstance(val, str):
            return self.mask_pii_and_si(val)
        elif isinstance(val, dict):
            return {k: self.mask_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self.mask_value(v) for v in val]
        elif isinstance(val, tuple):
            return tuple(self.mask_value(v) for v in val)
        elif isinstance(val, set):
            return {self.mask_value(v) for v in val}

        # Handle custom objects (like LangChain message classes)
        if hasattr(val, "content"):
            if hasattr(val, "copy"):
                try:
                    return val.copy(update={"content": self.mask_value(val.content)})
                except Exception:
                    pass
            try:
                # Direct mutation fallback
                val.content = self.mask_value(val.content)
                return val
            except Exception:
                pass

        return val

    def _extract_prompt_text(self, canonical_req: CanonicalRequest, args: tuple[Any, ...]) -> str:
        texts = []
        if canonical_req.messages:
            for msg in canonical_req.messages:
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("text") or ""
                    if isinstance(content, str) and content:
                        texts.append(content)
                elif isinstance(msg, str):
                    texts.append(msg)
                elif hasattr(msg, "content"):
                    c = getattr(msg, "content", "")
                    if isinstance(c, str) and c:
                        texts.append(c)

        if not texts and args:
            for arg in args:
                if isinstance(arg, str):
                    texts.append(arg)
                elif hasattr(arg, "content"):
                    c = getattr(arg, "content", "")
                    if isinstance(c, str) and c:
                        texts.append(c)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, str):
                            texts.append(item)
                        elif hasattr(item, "content"):
                            c = getattr(item, "content", "")
                            if isinstance(c, str) and c:
                                texts.append(c)

        return "\n".join(texts)

    def _process_input_guardrails(
        self, canonical_req: CanonicalRequest, args: tuple[Any, ...]
    ) -> tuple[bool, dict[str, Any] | None, CanonicalResponse | None]:
        if not hasattr(self, "guardrails_engine") or not self.guardrails_engine.is_enabled():
            return True, None, None

        prompt_text = self._extract_prompt_text(canonical_req, args)
        input_passed, input_rails_eval, input_violations = self.guardrails_engine.evaluate_input(prompt_text)

        if not input_passed:
            first_violation = input_violations[0]
            blocked_msg = f"[NeMo Guardrail Blocked] {first_violation['message']}"
            guardrail_meta = {
                "passed": False,
                "input_passed": False,
                "output_passed": True,
                "input_rails_evaluated": input_rails_eval,
                "output_rails_evaluated": [],
                "violations": input_violations,
                "blocked_by": first_violation["rail"],
                "message": first_violation["message"],
            }
            canonical_resp = CanonicalResponse(
                provider=canonical_req.provider,
                output=blocked_msg,
                model=canonical_req.model,
                latency=0.0,
                metadata={"guardrails": guardrail_meta},
            )
            return False, guardrail_meta, canonical_resp

        return True, {
            "input_passed": True,
            "input_rails_evaluated": input_rails_eval,
            "violations": []
        }, None

    def _process_output_guardrails(
        self, canonical_resp: CanonicalResponse, prompt_text: str, input_meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not hasattr(self, "guardrails_engine") or not self.guardrails_engine.is_enabled():
            return canonical_resp.metadata.get("guardrails", {})

        output_text = str(canonical_resp.output or "")
        output_passed, output_rails_eval, output_violations = self.guardrails_engine.evaluate_output(output_text, prompt_text)

        input_meta = input_meta or {}
        input_passed = input_meta.get("input_passed", True)
        input_rails_eval = input_meta.get("input_rails_evaluated", [])
        input_violations = input_meta.get("violations", [])

        combined_violations = input_violations + output_violations
        all_passed = input_passed and output_passed

        blocked_by = None
        message = None
        if not input_passed:
            blocked_by = input_violations[0]["rail"] if input_violations else None
            message = input_violations[0]["message"] if input_violations else None
        elif not output_passed:
            blocked_by = output_violations[0]["rail"] if output_violations else None
            message = output_violations[0]["message"] if output_violations else None
            canonical_resp.output = f"[NeMo Guardrail Blocked] {output_violations[0]['message']}"

        guardrail_meta = {
            "passed": all_passed,
            "input_passed": input_passed,
            "output_passed": output_passed,
            "input_rails_evaluated": input_rails_eval,
            "output_rails_evaluated": output_rails_eval,
            "violations": combined_violations,
            "blocked_by": blocked_by,
            "message": message,
        }
        canonical_resp.metadata["guardrails"] = guardrail_meta
        return guardrail_meta

    def handle_call(
        self,
        adapter: ProviderAdapter,
        make_live_call_fn: Callable[..., Any],
        *args: Any,
        is_stream: bool = False,
        is_parse: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Intercepts the call, checking the cache according to the execution mode."""
        if self.mask_pii:
            args = tuple(self.mask_value(arg) for arg in args)
            kwargs = {k: self.mask_value(v) for k, v in kwargs.items()}

        # 1. Convert request args to CanonicalRequest
        canonical_req = adapter.to_canonical_request(args, **kwargs)

        # Apply custom normalizer if configured
        if self.normalizer is not None:
            normalized_dict = self.normalizer(
                {
                    "provider": canonical_req.provider,
                    "model": canonical_req.model,
                    "messages": canonical_req.messages,
                    "temperature": canonical_req.temperature,
                    "params": canonical_req.params,
                }
            )
            # Rebuild canonical request from normalized dict
            canonical_req = CanonicalRequest(
                provider=normalized_dict.get("provider", canonical_req.provider),
                model=normalized_dict.get("model", canonical_req.model),
                messages=normalized_dict.get("messages", canonical_req.messages),
                temperature=normalized_dict.get("temperature", canonical_req.temperature),
                params=normalized_dict.get("params", canonical_req.params),
                raw=canonical_req.raw,
                metadata=canonical_req.metadata,
            )

        # 2. Hash request
        req_hash = hash_request(canonical_req, self.ignore_fields)
        cassette_path = self.find_cassette_path(req_hash, canonical_req.provider)
        save_path = self.get_cassette_path(req_hash, canonical_req.provider)

        prompt_text = self._extract_prompt_text(canonical_req, args)

        # 3. Evaluate Input Guardrails
        input_ok, input_meta, blocked_resp = self._process_input_guardrails(canonical_req, args)
        if not input_ok and blocked_resp is not None:
            if self.mode in ("record", "auto"):
                serialized_req = self._serialize_canonical_request(canonical_req)
                serialized_resp = self._serialize_canonical_response(blocked_resp)
                cassette_obj = Cassette(
                    provider=canonical_req.provider,
                    hash=req_hash,
                    request=serialized_req,
                    response=serialized_resp,
                    metadata={"latency_ms": 0.0, "guardrails": blocked_resp.metadata.get("guardrails")},
                )
                storage.save(cassette_obj, save_path, base_dir=self.path)
            return adapter.from_canonical_response(blocked_resp, args, is_parse=is_parse, **kwargs)

        # Mode: live
        if self.mode == "live":
            live_response = make_live_call_fn(*args, **kwargs)
            canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
            guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
            if guardrail_meta and not guardrail_meta.get("output_passed", True):
                return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)
            return live_response

        # Mode: replay
        if self.mode == "replay":
            if not storage.exists(cassette_path):
                raise CassetteNotFoundError(
                    f"Cassette not found in replay mode at path: {cassette_path}"
                )
            cassette_obj = storage.load(cassette_path)
            
            is_stored_stream = cassette_obj.response.get("metadata", {}).get("is_stream", False)
            if is_stored_stream or is_stream:
                chunks_data = cassette_obj.response.get("metadata", {}).get("chunks", [])
                
                def gen():
                    for chunk_data in chunks_data:
                        yield deserialize_chunk(chunk_data)
                
                return ReplayStream(gen())

            canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
            self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
            return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        # Mode: auto
        if self.mode == "auto":
            if storage.exists(cassette_path):
                cassette_obj = storage.load(cassette_path)
                is_stored_stream = cassette_obj.response.get("metadata", {}).get("is_stream", False)
                if is_stored_stream or is_stream:
                    chunks_data = cassette_obj.response.get("metadata", {}).get("chunks", [])
                    
                    def gen():
                        for chunk_data in chunks_data:
                            yield deserialize_chunk(chunk_data)
                    
                    return ReplayStream(gen())

                canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
                self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
                return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        # Mode: record, or auto on cache miss
        if is_stream:
            start_time = time.perf_counter()
            live_stream = make_live_call_fn(*args, **kwargs)
            
            def record_gen():
                chunks = []
                if hasattr(live_stream, "__enter__"):
                    with live_stream as stream:
                        for chunk in stream:
                            if self.mask_pii:
                                chunk = self.mask_value(chunk)
                            chunks.append(serialize_chunk(chunk))
                            yield chunk
                else:
                    for chunk in live_stream:
                        if self.mask_pii:
                            chunk = self.mask_value(chunk)
                        chunks.append(serialize_chunk(chunk))
                        yield chunk
                
                latency = (time.perf_counter() - start_time) * 1000.0
                full_output = "".join(extract_text_from_chunk(deserialize_chunk(c)) for c in chunks)
                
                canonical_resp = CanonicalResponse(
                    provider=canonical_req.provider,
                    output=full_output,
                    model=canonical_req.model,
                    latency=latency,
                    metadata={
                        "is_stream": True,
                        "chunks": chunks,
                    }
                )
                guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
                
                serialized_req = self._serialize_canonical_request(canonical_req)
                serialized_resp = self._serialize_canonical_response(canonical_resp)
                
                cassette_obj = Cassette(
                    provider=canonical_req.provider,
                    hash=req_hash,
                    request=serialized_req,
                    response=serialized_resp,
                    metadata={"latency_ms": latency, "guardrails": guardrail_meta},
                )
                storage.save(cassette_obj, save_path, base_dir=self.path)
                
            return ReplayStream(record_gen())

        # Non-stream record flow
        start_time = time.perf_counter()
        live_response = make_live_call_fn(*args, **kwargs)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms

        canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
        if canonical_resp.latency is None:
            canonical_resp.latency = latency

        guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)

        # Serialize request/response and save to storage
        serialized_req = self._serialize_canonical_request(canonical_req)
        serialized_resp = self._serialize_canonical_response(canonical_resp)

        cassette_obj = Cassette(
            provider=canonical_req.provider,
            hash=req_hash,
            request=serialized_req,
            response=serialized_resp,
            metadata={"latency_ms": latency, "guardrails": guardrail_meta},
        )
        storage.save(cassette_obj, save_path, base_dir=self.path)

        if guardrail_meta and not guardrail_meta.get("output_passed", True):
            return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        return live_response

    async def handle_call_async(
        self,
        adapter: ProviderAdapter,
        make_live_call_fn: Callable[..., Any],
        *args: Any,
        is_stream: bool = False,
        is_parse: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Intercepts the async call, checking the cache according to the execution mode."""
        if self.mask_pii:
            args = tuple(self.mask_value(arg) for arg in args)
            kwargs = {k: self.mask_value(v) for k, v in kwargs.items()}

        # 1. Convert request args to CanonicalRequest
        canonical_req = adapter.to_canonical_request(args, **kwargs)

        # Apply custom normalizer if configured
        if self.normalizer is not None:
            normalized_dict = self.normalizer(
                {
                    "provider": canonical_req.provider,
                    "model": canonical_req.model,
                    "messages": canonical_req.messages,
                    "temperature": canonical_req.temperature,
                    "params": canonical_req.params,
                }
            )
            # Rebuild canonical request from normalized dict
            canonical_req = CanonicalRequest(
                provider=normalized_dict.get("provider", canonical_req.provider),
                model=normalized_dict.get("model", canonical_req.model),
                messages=normalized_dict.get("messages", canonical_req.messages),
                temperature=normalized_dict.get("temperature", canonical_req.temperature),
                params=normalized_dict.get("params", canonical_req.params),
                raw=canonical_req.raw,
                metadata=canonical_req.metadata,
            )

        # 2. Hash request
        req_hash = hash_request(canonical_req, self.ignore_fields)
        cassette_path = self.find_cassette_path(req_hash, canonical_req.provider)
        save_path = self.get_cassette_path(req_hash, canonical_req.provider)

        prompt_text = self._extract_prompt_text(canonical_req, args)

        # 3. Evaluate Input Guardrails
        input_ok, input_meta, blocked_resp = self._process_input_guardrails(canonical_req, args)
        if not input_ok and blocked_resp is not None:
            if self.mode in ("record", "auto"):
                serialized_req = self._serialize_canonical_request(canonical_req)
                serialized_resp = self._serialize_canonical_response(blocked_resp)
                cassette_obj = Cassette(
                    provider=canonical_req.provider,
                    hash=req_hash,
                    request=serialized_req,
                    response=serialized_resp,
                    metadata={"latency_ms": 0.0, "guardrails": blocked_resp.metadata.get("guardrails")},
                )
                storage.save(cassette_obj, save_path, base_dir=self.path)
            return adapter.from_canonical_response(blocked_resp, args, is_parse=is_parse, **kwargs)

        # Mode: live
        if self.mode == "live":
            live_response = await make_live_call_fn(*args, **kwargs)
            canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
            guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
            if guardrail_meta and not guardrail_meta.get("output_passed", True):
                return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)
            return live_response

        # Mode: replay
        if self.mode == "replay":
            if not storage.exists(cassette_path):
                raise CassetteNotFoundError(
                    f"Cassette not found in replay mode at path: {cassette_path}"
                )
            cassette_obj = storage.load(cassette_path)
            
            is_stored_stream = cassette_obj.response.get("metadata", {}).get("is_stream", False)
            if is_stored_stream or is_stream:
                chunks_data = cassette_obj.response.get("metadata", {}).get("chunks", [])
                
                async def async_gen():
                    for chunk_data in chunks_data:
                        yield deserialize_chunk(chunk_data)
                
                return ReplayAsyncStream(async_gen())

            canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
            self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
            return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        # Mode: auto
        if self.mode == "auto":
            if storage.exists(cassette_path):
                cassette_obj = storage.load(cassette_path)
                is_stored_stream = cassette_obj.response.get("metadata", {}).get("is_stream", False)
                if is_stored_stream or is_stream:
                    chunks_data = cassette_obj.response.get("metadata", {}).get("chunks", [])
                    
                    async def async_gen():
                        for chunk_data in chunks_data:
                            yield deserialize_chunk(chunk_data)
                    
                    return ReplayAsyncStream(async_gen())

                canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
                self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
                return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        # Mode: record, or auto on cache miss
        if is_stream:
            start_time = time.perf_counter()
            live_stream = await make_live_call_fn(*args, **kwargs)
            
            async def record_async_gen():
                chunks = []
                if hasattr(live_stream, "__aenter__"):
                    async with live_stream as stream:
                        async for chunk in stream:
                            if self.mask_pii:
                                chunk = self.mask_value(chunk)
                            chunks.append(serialize_chunk(chunk))
                            yield chunk
                else:
                    async for chunk in live_stream:
                        if self.mask_pii:
                            chunk = self.mask_value(chunk)
                        chunks.append(serialize_chunk(chunk))
                        yield chunk
                
                latency = (time.perf_counter() - start_time) * 1000.0
                full_output = "".join(extract_text_from_chunk(deserialize_chunk(c)) for c in chunks)
                
                canonical_resp = CanonicalResponse(
                    provider=canonical_req.provider,
                    output=full_output,
                    model=canonical_req.model,
                    latency=latency,
                    metadata={
                        "is_stream": True,
                        "chunks": chunks,
                    }
                )
                guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)
                
                serialized_req = self._serialize_canonical_request(canonical_req)
                serialized_resp = self._serialize_canonical_response(canonical_resp)
                
                cassette_obj = Cassette(
                    provider=canonical_req.provider,
                    hash=req_hash,
                    request=serialized_req,
                    response=serialized_resp,
                    metadata={"latency_ms": latency, "guardrails": guardrail_meta},
                )
                storage.save(cassette_obj, save_path, base_dir=self.path)
                
            return ReplayAsyncStream(record_async_gen())

        # Non-stream record flow
        start_time = time.perf_counter()
        live_response = await make_live_call_fn(*args, **kwargs)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms

        canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
        if canonical_resp.latency is None:
            canonical_resp.latency = latency

        guardrail_meta = self._process_output_guardrails(canonical_resp, prompt_text, input_meta)

        # Serialize request/response and save to storage
        serialized_req = self._serialize_canonical_request(canonical_req)
        serialized_resp = self._serialize_canonical_response(canonical_resp)

        cassette_obj = Cassette(
            provider=canonical_req.provider,
            hash=req_hash,
            request=serialized_req,
            response=serialized_resp,
            metadata={"latency_ms": latency, "guardrails": guardrail_meta},
        )
        storage.save(cassette_obj, save_path, base_dir=self.path)

        if guardrail_meta and not guardrail_meta.get("output_passed", True):
            return adapter.from_canonical_response(canonical_resp, args, is_parse=is_parse, **kwargs)

        return live_response

    def _serialize_canonical_request(self, req: CanonicalRequest) -> dict[str, Any]:
        return {
            "provider": req.provider,
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "params": req.params,
        }

    def _serialize_canonical_response(self, resp: CanonicalResponse) -> dict[str, Any]:
        return {
            "provider": resp.provider,
            "output": resp.output,
            "model": resp.model,
            "tool_calls": resp.tool_calls,
            "invalid_tool_calls": resp.invalid_tool_calls,
            "latency": resp.latency,
            "reasoning": resp.reasoning,
            "usage": resp.usage,
            "metadata": resp.metadata,
        }

    def _deserialize_canonical_response(self, data: dict[str, Any]) -> CanonicalResponse:
        return CanonicalResponse(
            provider=data.get("provider", ""),
            output=data.get("output"),
            model=data.get("model"),
            tool_calls=data.get("tool_calls") or [],
            invalid_tool_calls=data.get("invalid_tool_calls") or [],
            latency=data.get("latency"),
            reasoning=data.get("reasoning"),
            usage=data.get("usage"),
            metadata=data.get("metadata") or {},
        )
