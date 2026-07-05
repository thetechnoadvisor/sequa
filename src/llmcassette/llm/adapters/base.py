from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalRequest:
    provider: str
    operation: str
    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalResponse:
    provider: str
    operation: str
    model: str | None = None
    output: Any = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    provider_name: str = ""

    @abstractmethod
    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        """Convert a provider-specific request into the canonical request shape."""

    @abstractmethod
    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        """Convert a provider-specific response into the canonical response shape."""

    @abstractmethod
    def from_canonical_response(self, response: CanonicalResponse, request: Any) -> Any:
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
            raise KeyError(f"No adapter registered for provider: {provider_name}") from exc