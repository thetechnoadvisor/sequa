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
        messages = payload.get("messages", [])
        return CanonicalRequest(
            provider=self.provider_name,
            model=kwargs.get("model"),
            messages=list(messages),
            params=kwargs,
        )

    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        payload = response or {}
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
        if raw_resp and isinstance(raw_resp, dict):
            resp_id = raw_resp.get("id", "replayed-response")
            choices = raw_resp.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason", "stop")

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
