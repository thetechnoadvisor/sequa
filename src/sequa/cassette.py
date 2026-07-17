from __future__ import annotations

import inspect
import threading
from functools import wraps
from typing import Any, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from sequa.recorder import RecorderEngine

F = TypeVar("F", bound=Callable[..., Any])

_local = threading.local()


def get_active_engine() -> Any | None:
    """Retrieve the currently active RecorderEngine from the thread-local stack."""
    if not hasattr(_local, "stack"):
        _local.stack = []
    if _local.stack:
        return _local.stack[-1]
    return None


class cassette:
    def __init__(
        self,
        path: str = "cassettes",
        mode: str = "auto",
        ignore_fields: list[str] | None = None,
        normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        adapter: Any | None = None,
        mask_pii: bool = False,
    ) -> None:
        self.path = path
        self.mode = mode
        self.ignore_fields = ignore_fields or []
        self.normalizer = normalizer
        self.adapter = adapter
        self.mask_pii = mask_pii

        if self.adapter is None:
            from sequa.llm.adapters.chat import LangChainGroqAdapter

            self.adapter = LangChainGroqAdapter()

        from sequa.recorder import RecorderEngine
        self.engine = RecorderEngine(
            path=self.path,
            mode=self.mode,
            ignore_fields=self.ignore_fields,
            normalizer=self.normalizer,
            mask_pii=self.mask_pii,
        )
        self.original_methods: list[tuple[type, str, Any]] = []
        self.patchers: list[Any] = []

    def __enter__(self) -> cassette:
        # 1. Push self.engine onto thread-local context stack
        if not hasattr(_local, "stack"):
            _local.stack = []
        _local.stack.append(self.engine)

        # 2. Patch langchain_groq ChatGroq if installed
        try:
            from langchain_groq import ChatGroq

            # Sync invoke
            if not getattr(ChatGroq.invoke, "__sequa_patched__", False):
                original_invoke = ChatGroq.invoke
                self.original_methods.append((ChatGroq, "invoke", original_invoke))

                def wrapped_invoke(self_obj: ChatGroq, *args: Any, **kwargs: Any) -> Any:
                    active_engine = get_active_engine()
                    if active_engine is None:
                        return original_invoke(self_obj, *args, **kwargs)

                    def make_live_call(s: Any, *a: Any, **kw: Any) -> Any:
                        return original_invoke(s, *a, **kw)

                    return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, **kwargs)

                wrapped_invoke.__sequa_patched__ = True
                ChatGroq.invoke = wrapped_invoke

            # Async ainvoke
            if hasattr(ChatGroq, "ainvoke"):
                if not getattr(ChatGroq.ainvoke, "__sequa_patched__", False):
                    original_ainvoke = ChatGroq.ainvoke
                    self.original_methods.append((ChatGroq, "ainvoke", original_ainvoke))

                    async def wrapped_ainvoke(self_obj: ChatGroq, *args: Any, **kwargs: Any) -> Any:
                        active_engine = get_active_engine()
                        if active_engine is None:
                            return await original_ainvoke(self_obj, *args, **kwargs)

                        async def make_live_call(s: Any, *a: Any, **kw: Any) -> Any:
                            return await original_ainvoke(s, *a, **kw)

                        return await active_engine.handle_call_async(
                            self.adapter, make_live_call, self_obj, *args, **kwargs
                        )

                    wrapped_ainvoke.__sequa_patched__ = True
                    ChatGroq.ainvoke = wrapped_ainvoke

            # Sync stream
            if hasattr(ChatGroq, "stream"):
                if not getattr(ChatGroq.stream, "__sequa_patched__", False):
                    original_stream = ChatGroq.stream
                    self.original_methods.append((ChatGroq, "stream", original_stream))

                    def wrapped_stream(self_obj: ChatGroq, *args: Any, **kwargs: Any) -> Any:
                        active_engine = get_active_engine()
                        if active_engine is None:
                            return original_stream(self_obj, *args, **kwargs)

                        def make_live_call(s: Any, *a: Any, **kw: Any) -> Any:
                            return original_stream(s, *a, **kw)

                        return active_engine.handle_call(
                            self.adapter, make_live_call, self_obj, *args, is_stream=True, **kwargs
                        )

                    wrapped_stream.__sequa_patched__ = True
                    ChatGroq.stream = wrapped_stream

            # Async astream
            if hasattr(ChatGroq, "astream"):
                if not getattr(ChatGroq.astream, "__sequa_patched__", False):
                    original_astream = ChatGroq.astream
                    self.original_methods.append((ChatGroq, "astream", original_astream))

                    def wrapped_astream(self_obj: ChatGroq, *args: Any, **kwargs: Any) -> Any:
                        active_engine = get_active_engine()
                        if active_engine is None:
                            return original_astream(self_obj, *args, **kwargs)

                        async def stream_generator():
                            async def make_live_call(s: Any, *a: Any, **kw: Any) -> Any:
                                return original_astream(s, *a, **kw)

                            replay_stream = await active_engine.handle_call_async(
                                self.adapter, make_live_call, self_obj, *args, is_stream=True, **kwargs
                            )
                            async for chunk in replay_stream:
                                yield chunk

                        return stream_generator()

                    wrapped_astream.__sequa_patched__ = True
                    ChatGroq.astream = wrapped_astream

        except ImportError:
            pass

        # 3. Patch OpenAI if installed
        try:
            import openai
            from sequa.llm.adapters.patch_openai import OpenAIMonkeyPatch
            op_patcher = OpenAIMonkeyPatch()
            op_patcher.patch()
            self.patchers.append(op_patcher)
        except ImportError:
            pass

        # 4. Patch Anthropic if installed
        try:
            import anthropic
            from sequa.llm.adapters.patch_anthropic import AnthropicMonkeyPatch
            ant_patcher = AnthropicMonkeyPatch()
            ant_patcher.patch()
            self.patchers.append(ant_patcher)
        except ImportError:
            pass

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # 1. Pop from stack
        if hasattr(_local, "stack") and _local.stack:
            _local.stack.pop()

        # 2. Restore original methods
        for cls, attr, original in self.original_methods:
            setattr(cls, attr, original)
        self.original_methods.clear()

        # 3. Restore OpenAI and Anthropic patchers
        for patcher in self.patchers:
            patcher.restore()
        self.patchers.clear()

    def intercept(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Provides direct interception utility for testing."""
        canonical_request = self.adapter.to_canonical_request(
            request=args[0] if args else None, **kwargs
        )
        result = fn(*args, **kwargs)
        canonical_response = self.adapter.to_canonical_response(result, **kwargs)
        return {
            "request": canonical_request,
            "response": canonical_response,
            "raw": result,
        }

    def __call__(self, fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is not None:
                    async def make_live_call(*a: Any, **kw: Any) -> Any:
                        return await fn(*a, **kw)
                    return await active_engine.handle_call_async(
                        self.adapter, make_live_call, *args, **kwargs
                    )
                else:
                    canonical_request = self.adapter.to_canonical_request(args, **kwargs)
                    result = await fn(*args, **kwargs)
                    canonical_response = self.adapter.to_canonical_response(result, **kwargs)
                    return {
                        "request": canonical_request,
                        "response": canonical_response,
                        "raw": result,
                    }
            wrapped_async.__sequa_patched__ = True
            return wrapped_async  # type: ignore
        else:
            @wraps(fn)
            def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is not None:
                    def make_live_call(*a: Any, **kw: Any) -> Any:
                        return fn(*a, **kw)
                    return active_engine.handle_call(
                        self.adapter, make_live_call, *args, **kwargs
                    )
                else:
                    canonical_request = self.adapter.to_canonical_request(args, **kwargs)
                    result = fn(*args, **kwargs)
                    canonical_response = self.adapter.to_canonical_response(result, **kwargs)
                    return {
                        "request": canonical_request,
                        "response": canonical_response,
                        "raw": result,
                    }
            wrapped_sync.__sequa_patched__ = True
            return wrapped_sync  # type: ignore
