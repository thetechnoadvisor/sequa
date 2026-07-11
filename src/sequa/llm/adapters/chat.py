from __future__ import annotations

from typing import Any

from .base import CanonicalRequest, CanonicalResponse, ProviderAdapter


class LangChainGroqAdapter(ProviderAdapter):
    """A generic chat-completions adapter that can be configured per provider."""

    def __init__(
        self, provider_name: str = "langchain_groq", response_object: str | None = None
    ) -> None:
        self.provider_name = provider_name
        self._response_object = response_object

    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        if isinstance(request, (tuple, list)) and len(request) > 0:
            first_arg = request[0]
            if hasattr(first_arg, "model_name"):
                payload = first_arg
                messages = request[1] if len(request) > 1 else []
            else:
                payload = first_arg
                messages = []
        else:
            payload = request or {}
            messages = []

        model = getattr(payload, "model_name", None) or kwargs.get("model")
        temperature = getattr(payload, "temperature", None) or kwargs.get("temperature")

        if isinstance(payload, dict):
            raw_payload = payload
            if not messages:
                messages = payload.get("messages") or kwargs.get("messages") or []
        else:
            raw_payload = {"input": payload}
            if not messages:
                messages = kwargs.get("messages") or []

        if isinstance(messages, (list, tuple)):
            messages_list = list(messages)
        elif messages:
            messages_list = [messages]
        else:
            messages_list = []

        return CanonicalRequest(
            provider=self.provider_name,
            model=model,
            temperature=temperature,
            messages=messages_list,
            params=kwargs,
            metadata={"raw_request": raw_payload},
        )

    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        payload = response or {}
        if hasattr(payload, "content") and not isinstance(payload, dict):
            content = getattr(payload, "content", None)
            return CanonicalResponse(
                provider=self.provider_name,
                output=content,
                model=payload.response_metadata.get("model_name"),
                tool_calls=payload.tool_calls,
                invalid_tool_calls=payload.invalid_tool_calls,
                latency=payload.response_metadata.get("total_time"),
                reasoning=(
                    payload.response_metadata.get("token_usage", {}).get("reasoning_content")
                    if payload.response_metadata.get("token_usage")
                    else None
                ),
                usage=payload.usage_metadata,
                metadata={
                    "raw_response": {
                        "id": getattr(payload, "id", None),
                        "response_metadata": getattr(payload, "response_metadata", {}),
                    }
                },
            )

        choices = payload.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})

        return CanonicalResponse(
            provider=self.provider_name,
            model=kwargs.get("model"),
            output=message.get("content"),
            usage=payload.get("usage"),
            metadata={"raw_response": payload},
        )

    def from_canonical_response(self, response: CanonicalResponse, request: Any) -> Any:
        raw_resp = response.metadata.get("raw_response")
        resp_id = "replayed-response"
        finish_reason = "stop"
        if raw_resp:
            if isinstance(raw_resp, dict):
                resp_id = raw_resp.get("id", "replayed-response")
                choices = raw_resp.get("choices", [])
                if choices:
                    finish_reason = choices[0].get("finish_reason", "stop")
            else:
                resp_id = getattr(raw_resp, "id", "replayed-response")
                response_metadata = getattr(raw_resp, "response_metadata", {})
                finish_reason = response_metadata.get("finish_reason", "stop")

        # Determine if we should return a LangChain AIMessage object
        is_langchain = False
        if request and isinstance(request, (tuple, list)) and len(request) > 0:
            first_arg = request[0]
            # If the class represents ChatGroq or a LangChain chat model
            if hasattr(first_arg, "model_name") or hasattr(first_arg, "invoke"):
                is_langchain = True

        if raw_resp and hasattr(raw_resp, "content") and not isinstance(raw_resp, dict):
            is_langchain = True

        if is_langchain:
            from langchain_core.messages import AIMessage
            
            response_metadata = {}
            if raw_resp:
                if isinstance(raw_resp, dict):
                    response_metadata = dict(raw_resp.get("response_metadata", {}))
                elif hasattr(raw_resp, "response_metadata"):
                    response_metadata = dict(getattr(raw_resp, "response_metadata", {}))

            if not response_metadata:
                response_metadata = {
                    "model_name": response.model,
                    "finish_reason": finish_reason,
                }
                if response.latency is not None:
                    response_metadata["total_time"] = response.latency

            aimsg_kwargs: dict[str, Any] = {
                "content": response.output or "",
                "response_metadata": response_metadata,
                "id": resp_id,
                "tool_calls": response.tool_calls or [],
                "invalid_tool_calls": response.invalid_tool_calls or [],
            }

            if response.usage and isinstance(response.usage, dict):
                input_t = response.usage.get("input_tokens") or response.usage.get("prompt_tokens") or 0
                output_t = response.usage.get("output_tokens") or response.usage.get("completion_tokens") or 0
                total_t = response.usage.get("total_tokens") or (input_t + output_t)
                aimsg_kwargs["usage_metadata"] = {
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": total_t,
                }

            return AIMessage(**aimsg_kwargs)

        payload: dict[str, Any] = {
            "id": resp_id,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.output,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": response.usage or {},
            "model": response.model or "replayed-model",
        }

        if self._response_object is not None:
            payload["object"] = self._response_object

        return payload


class OpenAIAdapter(ProviderAdapter):
    """An OpenAI chat completions adapter."""

    def __init__(self, response_object: str | None = "chat.completion") -> None:
        self.provider_name = "openai"
        self._response_object = response_object

    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        payload = request or {}
        if not isinstance(payload, dict):
            if isinstance(payload, (list, tuple)) and len(payload) > 0:
                payload = payload[0] if isinstance(payload[0], dict) else {}
            else:
                payload = {}

        model = kwargs.get("model") or payload.get("model")
        messages = kwargs.get("messages") or payload.get("messages") or []
        temperature = kwargs.get("temperature") or payload.get("temperature")
        
        params = {k: v for k, v in kwargs.items() if k not in ("model", "messages")}
        for k, v in payload.items():
            if k not in ("model", "messages") and k not in params:
                params[k] = v
        
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append({"role": msg.get("role"), "content": msg.get("content")})
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                normalized_messages.append({"role": role, "content": content})

        return CanonicalRequest(
            provider=self.provider_name,
            model=model,
            temperature=temperature,
            messages=normalized_messages,
            params=params,
            metadata={"raw_request": kwargs},
        )

    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        if isinstance(response, dict):
            payload = response
            choices = payload.get("choices", [])
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message", {})
            output = message.get("content")
            model = payload.get("model") or kwargs.get("model")
            usage = payload.get("usage")
            resp_id = payload.get("id")
            tool_calls = message.get("tool_calls") or []
        else:
            # It's a ChatCompletion object
            resp_id = getattr(response, "id", None)
            model = getattr(response, "model", None) or kwargs.get("model")
            usage_obj = getattr(response, "usage", None)
            usage = None
            if usage_obj:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": getattr(usage_obj, "total_tokens", 0),
                }
            
            choices = getattr(response, "choices", [])
            first_choice = choices[0] if choices else None
            output = None
            tool_calls = []
            if first_choice:
                message_obj = getattr(first_choice, "message", None)
                if message_obj:
                    output = getattr(message_obj, "content", None)
                    tool_calls = getattr(message_obj, "tool_calls", None) or []

        # Convert choices list to serializable dictionary format
        choices_serializable = []
        for c in choices:
            if isinstance(c, dict):
                msg = c.get("message") or {}
                choices_serializable.append({
                    "finish_reason": c.get("finish_reason", "stop"),
                    "message": {
                        "role": msg.get("role", "assistant"),
                        "content": msg.get("content", ""),
                    }
                })
            else:
                msg_obj = getattr(c, "message", None)
                choices_serializable.append({
                    "finish_reason": getattr(c, "finish_reason", "stop"),
                    "message": {
                        "role": getattr(msg_obj, "role", "assistant") if msg_obj else "assistant",
                        "content": getattr(msg_obj, "content", "") if msg_obj else "",
                    }
                })

        return CanonicalResponse(
            provider=self.provider_name,
            model=model,
            output=output,
            usage=usage,
            tool_calls=list(tool_calls),
            metadata={
                "raw_response": {
                    "id": resp_id,
                    "model": model,
                    "choices": choices_serializable,
                }
            }
        )

    def from_canonical_response(self, response: CanonicalResponse, request: Any) -> Any:
        raw_resp = response.metadata.get("raw_response", {})
        resp_id = raw_resp.get("id") or "replayed-response"
        choices_data = raw_resp.get("choices") or []
        
        choices_list = []
        for i, choice in enumerate(choices_data):
            msg_data = choice.get("message", {})
            choices_list.append({
                "index": i,
                "finish_reason": choice.get("finish_reason", "stop"),
                "message": {
                    "role": msg_data.get("role", "assistant"),
                    "content": msg_data.get("content", response.output),
                }
            })
            
        usage_data = response.usage or {}

        try:
            from openai.types.chat import ChatCompletion
            from openai.types.chat.chat_completion import Choice, ChoiceMessage
            from openai.types import CompletionUsage
            
            choices = []
            for item in choices_list:
                msg = item["message"]
                choices.append(Choice(
                    finish_reason=item["finish_reason"],
                    index=item["index"],
                    message=ChoiceMessage(
                        content=msg["content"],
                        role=msg["role"],
                    )
                ))
            
            usage = None
            if usage_data:
                usage = CompletionUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
                
            return ChatCompletion(
                id=resp_id,
                choices=choices,
                created=123456789,
                model=response.model or "replayed-model",
                object="chat.completion",
                usage=usage,
            )
        except ImportError:
            class MockMessage:
                def __init__(self, role: str, content: str):
                    self.role = role
                    self.content = content
                    
            class MockChoice:
                def __init__(self, index: int, finish_reason: str, message: MockMessage):
                    self.index = index
                    self.finish_reason = finish_reason
                    self.message = message
                    
            class MockUsage:
                def __init__(self, prompt: int, completion: int, total: int):
                    self.prompt_tokens = prompt
                    self.completion_tokens = completion
                    self.total_tokens = total
                    
            class MockChatCompletion:
                def __init__(self, id: str, choices: list[MockChoice], model: str, usage: MockUsage | None):
                    self.id = id
                    self.choices = choices
                    self.model = model
                    self.usage = usage
                    self.object = "chat.completion"
                    
            choices = []
            for item in choices_list:
                msg = item["message"]
                choices.append(MockChoice(
                    index=item["index"],
                    finish_reason=item["finish_reason"],
                    message=MockMessage(role=msg["role"], content=msg["content"])
                ))
                
            usage = None
            if usage_data:
                usage = MockUsage(
                    usage_data.get("prompt_tokens") or usage_data.get("input_tokens") or 0,
                    usage_data.get("completion_tokens") or usage_data.get("output_tokens") or 0,
                    usage_data.get("total_tokens") or 0,
                )
                
            return MockChatCompletion(
                id=resp_id,
                choices=choices,
                model=response.model or "replayed-model",
                usage=usage,
            )
