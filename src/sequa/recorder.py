from __future__ import annotations

import os
import time
from typing import Any, Callable

from sequa.llm.adapters.base import CanonicalRequest, CanonicalResponse, ProviderAdapter
from sequa.matcher import hash_request
from sequa.models import Cassette
from sequa import storage


class SequaError(Exception):
    """Base exception for Sequa."""


class CassetteNotFoundError(SequaError):
    """Exception raised when a cassette is missing in replay mode."""


class RecorderEngine:
    def __init__(
        self,
        path: str,
        mode: str = "auto",
        ignore_fields: list[str] | None = None,
        normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.mode = mode.lower()
        self.ignore_fields = ignore_fields or []
        self.normalizer = normalizer

        if self.mode not in ("replay", "record", "auto", "live"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be replay, record, auto, or live.")

    def get_cassette_path(self, req_hash: str) -> str:
        """Resolve the path to the cassette file based on request hash and configured path."""
        # If the path is a directory (does not end in .json), append the hash filename
        if not self.path.lower().endswith(".json"):
            return os.path.join(self.path, f"{req_hash}.json")
        return self.path

    def handle_call(
        self,
        adapter: ProviderAdapter,
        make_live_call_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercepts the call, checking the cache according to the execution mode."""
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
        cassette_path = self.get_cassette_path(req_hash)

        # Mode: live
        if self.mode == "live":
            return make_live_call_fn(*args, **kwargs)

        # Mode: replay
        if self.mode == "replay":
            if not storage.exists(cassette_path):
                raise CassetteNotFoundError(
                    f"Cassette not found in replay mode at path: {cassette_path}"
                )
            cassette_obj = storage.load(cassette_path)
            canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
            return adapter.from_canonical_response(canonical_resp, args)

        # Mode: auto
        if self.mode == "auto":
            if storage.exists(cassette_path):
                cassette_obj = storage.load(cassette_path)
                canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
                return adapter.from_canonical_response(canonical_resp, args)

        # Mode: record, or auto on cache miss
        start_time = time.perf_counter()
        live_response = make_live_call_fn(*args, **kwargs)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms

        canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
        if canonical_resp.latency is None:
            canonical_resp.latency = latency

        # Serialize request/response and save to storage
        serialized_req = self._serialize_canonical_request(canonical_req)
        serialized_resp = self._serialize_canonical_response(canonical_resp)

        cassette_obj = Cassette(
            provider=canonical_req.provider,
            hash=req_hash,
            request=serialized_req,
            response=serialized_resp,
            metadata={"latency_ms": latency},
        )
        storage.save(cassette_obj, cassette_path)

        return live_response

    async def handle_call_async(
        self,
        adapter: ProviderAdapter,
        make_live_call_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercepts the async call, checking the cache according to the execution mode."""
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
        cassette_path = self.get_cassette_path(req_hash)

        # Mode: live
        if self.mode == "live":
            return await make_live_call_fn(*args, **kwargs)

        # Mode: replay
        if self.mode == "replay":
            if not storage.exists(cassette_path):
                raise CassetteNotFoundError(
                    f"Cassette not found in replay mode at path: {cassette_path}"
                )
            cassette_obj = storage.load(cassette_path)
            canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
            return adapter.from_canonical_response(canonical_resp, args)

        # Mode: auto
        if self.mode == "auto":
            if storage.exists(cassette_path):
                cassette_obj = storage.load(cassette_path)
                canonical_resp = self._deserialize_canonical_response(cassette_obj.response)
                return adapter.from_canonical_response(canonical_resp, args)

        # Mode: record, or auto on cache miss
        start_time = time.perf_counter()
        live_response = await make_live_call_fn(*args, **kwargs)
        latency = (time.perf_counter() - start_time) * 1000.0  # in ms

        canonical_resp = adapter.to_canonical_response(live_response, **kwargs)
        if canonical_resp.latency is None:
            canonical_resp.latency = latency

        # Serialize request/response and save to storage
        serialized_req = self._serialize_canonical_request(canonical_req)
        serialized_resp = self._serialize_canonical_response(canonical_resp)

        cassette_obj = Cassette(
            provider=canonical_req.provider,
            hash=req_hash,
            request=serialized_req,
            response=serialized_resp,
            metadata={"latency_ms": latency},
        )
        storage.save(cassette_obj, cassette_path)

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
