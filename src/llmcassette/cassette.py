from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from src.llmcassette.llm.adapters.base import CanonicalRequest, CanonicalResponse

F = TypeVar("F", bound=Callable[..., Any])


class cassette:
    def __init__(self, adapter: Any | None = None) -> None:
        self.adapter = adapter
        if self.adapter is None:
            from llmcassette.llm.adapters.chat import LangChainGroqAdapter

            self.adapter = LangChainGroqAdapter()

    def __call__(self, fn: F) -> F:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> tuple[CanonicalRequest, CanonicalResponse, Any]:
            canonical_request = self.adapter.to_canonical_request(request=args[0] if args else None, **kwargs)
            result = fn(*args, **kwargs)
            canonical_response = self.adapter.to_canonical_response(result, **kwargs)
            return canonical_request, canonical_response, result

        return wrapped

    def intercept(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[CanonicalRequest, CanonicalResponse, Any]:
        canonical_request = self.adapter.to_canonical_request(request=args[0] if args else None, **kwargs)
        result = fn(*args, **kwargs)
        canonical_response = self.adapter.to_canonical_response(result, **kwargs)
        return canonical_request, canonical_response, result
