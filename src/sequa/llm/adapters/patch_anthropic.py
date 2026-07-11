from __future__ import annotations

from typing import Any
from sequa.llm.adapters.anthropic import AnthropicAdapter
from sequa.cassette import get_active_engine


class AnthropicMonkeyPatch:
    def __init__(self) -> None:
        self.adapter = AnthropicAdapter()
        self.original_create = None
        self.original_async_create = None
        self.patched = False

    def patch(self) -> None:
        if self.patched:
            return

        try:
            from anthropic.resources.messages import Messages, AsyncMessages
            
            # Sync Patch
            self.original_create = Messages.create
            
            def wrapped_create(self_obj: Messages, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return self.original_create(self_obj, *args, **kwargs)
                
                def make_live_call(*a: Any, **kw: Any) -> Any:
                    return self.original_create(self_obj, *a, **kw)
                
                return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, **kwargs)

            wrapped_create.__llmcassette_patched__ = True
            Messages.create = wrapped_create
            
            # Async Patch
            self.original_async_create = AsyncMessages.create
            
            async def wrapped_async_create(self_obj: AsyncMessages, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return await self.original_async_create(self_obj, *args, **kwargs)
                
                async def make_live_call(*a: Any, **kw: Any) -> Any:
                    return await self.original_async_create(self_obj, *a, **kw)
                
                return await active_engine.handle_call_async(self.adapter, make_live_call, self_obj, *args, **kwargs)

            wrapped_async_create.__llmcassette_patched__ = True
            AsyncMessages.create = wrapped_async_create
            
            self.patched = True
        except ImportError:
            pass

    def restore(self) -> None:
        if not self.patched:
            return
        
        try:
            from anthropic.resources.messages import Messages, AsyncMessages
            if self.original_create:
                Messages.create = self.original_create
            if self.original_async_create:
                AsyncMessages.create = self.original_async_create
            self.patched = False
        except ImportError:
            pass
