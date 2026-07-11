from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq

from sequa.cassette import cassette
from sequa.llm.adapters.chat import LangChainGroqAdapter


def patch_langchain(adapter: Any | None = None) -> type[ChatGroq]:
    if getattr(ChatGroq.invoke, "__sequa_patched__", False):
        return ChatGroq

    adapter = adapter or LangChainGroqAdapter()
    decorator = cassette(adapter=adapter)
    original_invoke = ChatGroq.invoke

    @decorator
    def wrapped_invoke(self: ChatGroq, *args: Any, **kwargs: Any) -> Any:
        return original_invoke(self, *args, **kwargs)

    ChatGroq.invoke = wrapped_invoke
    return ChatGroq
