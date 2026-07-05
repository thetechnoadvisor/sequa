from __future__ import annotations

from typing import Any

from llmcassette.cassette import cassette


class LangChainGroqMonkeyPatch:
    def __init__(self, adapter: Any | None = None) -> None:
        self.adapter = adapter
        if self.adapter is None:
            from llmcassette.llm.adapters.chat import LangChainGroqAdapter

            self.adapter = LangChainGroqAdapter()
        self.decorator = cassette(adapter=self.adapter)

    def patch(self, cls: Any) -> Any:
        original_invoke = cls.invoke

        @self.decorator
        def wrapped_invoke(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            return original_invoke(self_obj, *args, **kwargs)

        cls.invoke = wrapped_invoke
        return cls
