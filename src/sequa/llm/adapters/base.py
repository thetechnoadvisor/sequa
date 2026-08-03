from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalRequest:
    provider: str
    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def system_message(self) -> str:
        """Extract developer system instructions / system persona from messages or params."""
        sys_parts: list[str] = []
        if self.messages:
            for msg in self.messages:
                if isinstance(msg, dict):
                    role = str(msg.get("role", "")).lower()
                    if role in ("system", "developer"):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = "\n".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in content
                            )
                        sys_parts.append(str(content))
        if not sys_parts and self.params:
            sys_val = self.params.get("system") or self.params.get("system_instruction") or self.params.get("system_prompt") or self.params.get("system_message")
            if sys_val:
                sys_parts.append(str(sys_val))
        return "\n".join(sys_parts).strip()

    @property
    def system_prompt(self) -> str:
        return self.system_message

    @property
    def user_message(self) -> str:
        """Extract user input query / message from messages or params."""
        usr_parts: list[str] = []
        if self.messages:
            for msg in self.messages:
                if isinstance(msg, dict):
                    role = str(msg.get("role", "")).lower()
                    if role in ("user", "human"):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = "\n".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in content
                            )
                        usr_parts.append(str(content))
        if not usr_parts and self.params:
            usr_val = self.params.get("prompt") or self.params.get("user_prompt") or self.params.get("user_message")
            if usr_val:
                usr_parts.append(str(usr_val))
        return "\n---\n".join(usr_parts).strip()

    @property
    def user_prompt(self) -> str:
        return self.user_message


@dataclass(slots=True)
class CanonicalResponse:
    provider: str
    output: Any = None
    model: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    invalid_tool_calls: list[Any] = field(default_factory=list)
    latency: float | None = None
    reasoning: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    provider_name: str = ""

    @abstractmethod
    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        """Convert a provider-specific request into the canonical request shape."""

    @abstractmethod
    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        """Convert a provider-specific response into the canonical response shape."""

    @abstractmethod
    def from_canonical_response(
        self,
        response: CanonicalResponse,
        request: Any,
        is_parse: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Rebuild a provider-native response from a canonical response."""


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.provider_name] = adapter

    def get(self, provider_name: str) -> ProviderAdapter:
        try:
            return self._adapters[provider_name]
        except KeyError as exc:
            raise KeyError(
                f"No adapter registered for provider: {provider_name}"
            ) from exc
