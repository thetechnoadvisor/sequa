from __future__ import annotations

from typing import Any
from sequa.llm.adapters.anthropic import AnthropicAdapter
from sequa.cassette import get_active_engine


class AnthropicMonkeyPatch:
    def __init__(self) -> None:
        self.adapter = AnthropicAdapter()
        self.original_create = None
        self.original_async_create = None
        self.original_stream = None
        self.original_async_stream = None
        self.patched = False

    def patch(self) -> None:
        if self.patched:
            return

        try:
            from anthropic.resources.messages import Messages, AsyncMessages
            
            # Sync Patch Create
            self.original_create = Messages.create
            
            def wrapped_create(self_obj: Messages, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return self.original_create(self_obj, *args, **kwargs)
                
                def make_live_call(*a: Any, **kw: Any) -> Any:
                    return self.original_create(self_obj, *a, **kw)
                
                is_stream = kwargs.get("stream") is True
                return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, is_stream=is_stream, **kwargs)

            wrapped_create.__llmcassette_patched__ = True
            Messages.create = wrapped_create
            
            # Async Patch Create
            self.original_async_create = AsyncMessages.create
            
            async def wrapped_async_create(self_obj: AsyncMessages, *args: Any, **kwargs: Any) -> Any:
                active_engine = get_active_engine()
                if active_engine is None:
                    return await self.original_async_create(self_obj, *args, **kwargs)
                
                async def make_live_call(*a: Any, **kw: Any) -> Any:
                    return await self.original_async_create(self_obj, *a, **kw)
                
                is_stream = kwargs.get("stream") is True
                return await active_engine.handle_call_async(self.adapter, make_live_call, self_obj, *args, is_stream=is_stream, **kwargs)

            wrapped_async_create.__llmcassette_patched__ = True
            AsyncMessages.create = wrapped_async_create

            # Sync Patch Stream
            if hasattr(Messages, "stream"):
                self.original_stream = Messages.stream
                
                def wrapped_stream(self_obj: Messages, *args: Any, **kwargs: Any) -> Any:
                    active_engine = get_active_engine()
                    if active_engine is None:
                        return self.original_stream(self_obj, *args, **kwargs)
                    
                    def make_live_call(*a: Any, **kw: Any) -> Any:
                        return self.original_stream(self_obj, *a, **kw)
                    
                    return active_engine.handle_call(self.adapter, make_live_call, self_obj, *args, is_stream=True, **kwargs)
                
                wrapped_stream.__llmcassette_patched__ = True
                Messages.stream = wrapped_stream

            # Async Patch Stream
            if hasattr(AsyncMessages, "stream"):
                self.original_async_stream = AsyncMessages.stream
                
                async def wrapped_async_stream(self_obj: AsyncMessages, *args: Any, **kwargs: Any) -> Any:
                    active_engine = get_active_engine()
                    if active_engine is None:
                        return await self.original_async_stream(self_obj, *args, **kwargs)
                    
                    async def make_live_call(*a: Any, **kw: Any) -> Any:
                        return await self.original_async_stream(self_obj, *a, **kw)
                    
                    return await active_engine.handle_call_async(self.adapter, make_live_call, self_obj, *args, is_stream=True, **kwargs)
                
                wrapped_async_stream.__llmcassette_patched__ = True
                AsyncMessages.stream = wrapped_async_stream
            
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
            if self.original_stream:
                Messages.stream = self.original_stream
            if self.original_async_stream:
                AsyncMessages.stream = self.original_async_stream
            self.patched = False
        except ImportError:
            pass
