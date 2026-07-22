from __future__ import annotations

from typing import Any
from sequa.llm.adapters.chat import OpenAIAdapter
from sequa.cassette import get_active_engine


class OpenAIMonkeyPatch:
    def __init__(self) -> None:
        self.adapter = OpenAIAdapter()
        self.original_create = None
        self.original_async_create = None
        self.original_parse = None
        self.original_async_parse = None
        self.patched = False

    def patch(self) -> None:
        if self.patched:
            return

        try:
            from openai.resources.chat.completions import Completions, AsyncCompletions
            
            # Sync Patch Create
            self.original_create = Completions.create
            
            def wrapped_create(self_obj: Completions, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return self.original_create(self_obj, *args, **kwargs)
                
                def make_live_call(*a: Any, **kw: Any) -> Any:
                    return self.original_create(self_obj, *a, **kw)
                
                is_stream = kwargs.get("stream") is True
                return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, is_stream=is_stream, **kwargs)

            wrapped_create.__llmcassette_patched__ = True
            Completions.create = wrapped_create
            
            # Async Patch Create
            self.original_async_create = AsyncCompletions.create
            
            async def wrapped_async_create(self_obj: AsyncCompletions, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return await self.original_async_create(self_obj, *args, **kwargs)
                
                async def make_live_call(*a: Any, **kw: Any) -> Any:
                    return await self.original_async_create(self_obj, *a, **kw)
                
                is_stream = kwargs.get("stream") is True
                return await active_engine.handle_call_async(self.adapter, make_live_call, self_obj, *args, is_stream=is_stream, **kwargs)

            wrapped_async_create.__llmcassette_patched__ = True
            AsyncCompletions.create = wrapped_async_create

            # Sync Patch Parse
            if hasattr(Completions, "parse"):
                self.original_parse = Completions.parse

                def wrapped_parse(self_obj: Completions, *args: Any, **kwargs: Any) -> Any:
                    active_engine = get_active_engine()
                    if active_engine is None:
                        return self.original_parse(self_obj, *args, **kwargs)

                    def make_live_call(*a: Any, **kw: Any) -> Any:
                        return self.original_parse(self_obj, *a, **kw)

                    return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, is_parse=True, **kwargs)

                wrapped_parse.__llmcassette_patched__ = True
                Completions.parse = wrapped_parse

            # Async Patch Parse
            if hasattr(AsyncCompletions, "parse"):
                self.original_async_parse = AsyncCompletions.parse

                async def wrapped_async_parse(self_obj: AsyncCompletions, *args: Any, **kwargs: Any) -> Any:
                    active_engine = get_active_engine()
                    if active_engine is None:
                        return await self.original_async_parse(self_obj, *args, **kwargs)

                    async def make_live_call(*a: Any, **kw: Any) -> Any:
                        return await self.original_async_parse(self_obj, *a, **kw)

                    return await active_engine.handle_call_async(self.adapter, make_live_call, self_obj, *args, is_parse=True, **kwargs)

                wrapped_async_parse.__llmcassette_patched__ = True
                AsyncCompletions.parse = wrapped_async_parse
            
            self.patched = True
        except ImportError:
            pass

    def restore(self) -> None:
        if not self.patched:
            return
        
        try:
            from openai.resources.chat.completions import Completions, AsyncCompletions
            if self.original_create:
                Completions.create = self.original_create
            if self.original_async_create:
                AsyncCompletions.create = self.original_async_create
            if self.original_parse:
                Completions.parse = self.original_parse
            if self.original_async_parse:
                AsyncCompletions.parse = self.original_async_parse
            self.patched = False
        except ImportError:
            pass
