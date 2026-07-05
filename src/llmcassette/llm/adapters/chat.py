from __future__ import annotations

from typing import Any

from .base import CanonicalRequest, CanonicalResponse, ProviderAdapter


class ChatCompletionsAdapter(ProviderAdapter):
    """A generic chat-completions adapter that can be configured per provider."""

    def __init__(self, provider_name: str, response_object: str | None = None) -> None:
        self.provider_name = provider_name
        self._response_object = response_object

    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        payload = request or {}
        if isinstance(payload, dict):
            messages = payload.get("messages") or kwargs.get("messages") or []
            raw_payload = payload
        else:
            messages = kwargs.get("messages") or []
            raw_payload = {"input": payload}

        params = dict(kwargs)
        params.pop("messages", None)
        params.pop("model", None)

        return CanonicalRequest(
            provider=self.provider_name,
            operation="chat.completions.create",
            model=kwargs.get("model"),
            messages=list(messages),
            params=params,
            metadata={"raw_request": raw_payload},
        )

    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        payload = response or {}
        if hasattr(payload, "content") and not isinstance(payload, dict):
            content = getattr(payload, "content", None)
            return CanonicalResponse(
                provider=self.provider_name,
                operation="chat.completions.create",
                model=kwargs.get("model"),
                output=content,
                usage=None,
                finish_reason=None,
                raw={"content": content},
            )

        choices = payload.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})

        return CanonicalResponse(
            provider=self.provider_name,
            operation="chat.completions.create",
            model=kwargs.get("model"),
            output=message.get("content"),
            usage=payload.get("usage"),
            finish_reason=first_choice.get("finish_reason"),
            raw=dict(payload),
        )

    def from_canonical_response(self, response: CanonicalResponse, request: Any) -> Any:
        payload: dict[str, Any] = {
            "id": response.raw.get("id", "replayed-response"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.output,
                    },
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": response.usage or {},
            "model": response.model or "replayed-model",
        }

        if self._response_object is not None:
            payload["object"] = self._response_object

        return payload


class OpenAIAdapter(ChatCompletionsAdapter):
    def __init__(self) -> None:
        super().__init__(provider_name="openai", response_object="chat.completion")


class LangChainGroqAdapter(ChatCompletionsAdapter):
    def __init__(self) -> None:
        super().__init__(provider_name="langchain_groq")
