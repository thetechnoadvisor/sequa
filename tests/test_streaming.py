from __future__ import annotations

import os
import sys
import tempfile
import pytest

# Setup OpenAI mock classes for streaming
class MockDelta:
    def __init__(self, content=None):
        self.content = content

class MockChoice:
    def __init__(self, content=None):
        self.delta = MockDelta(content)
        self.finish_reason = None
        self.index = 0

class MockChatCompletionChunk:
    def __init__(self, content=None):
        self.choices = [MockChoice(content)]

# Setup Anthropic mock classes for streaming
class MockAnthropicDelta:
    def __init__(self, text=None):
        self.text = text

class MockAnthropicMessageChunk:
    def __init__(self, text=None):
        self.delta = MockAnthropicDelta(text)

class MockLangChainChunk:
    def __init__(self, content=None):
        self.content = content


# We will patch the classes dynamically at the beginning of tests and restore at the end
_originals = {}

def setup_module():
    # 1. Patch OpenAI
    from openai.resources.chat.completions import Completions
    _originals["openai_create"] = Completions.create
    
    def mock_openai_create(self, *args, **kwargs):
        stream = kwargs.get("stream", False)
        if stream:
            def chunk_gen():
                yield MockChatCompletionChunk("Hello ")
                yield MockChatCompletionChunk("there!")
            return chunk_gen()
        else:
            # Match standard mock format from test_masking.py
            from test_masking import MockChoiceMessage, MockChoice as OriginalMockChoice, MockChatCompletion
            msg = MockChoiceMessage("assistant", "Mock response")
            return MockChatCompletion("msg-123", kwargs.get("model", "gpt-4"), [OriginalMockChoice(msg)])
    Completions.create = mock_openai_create

    # 2. Patch Anthropic
    from anthropic.resources.messages import Messages
    _originals["anthropic_create"] = Messages.create
    
    def mock_anthropic_create(self, *args, **kwargs):
        stream = kwargs.get("stream", False)
        if stream:
            def chunk_gen():
                yield MockAnthropicMessageChunk("Anthropic ")
                yield MockAnthropicMessageChunk("stream!")
            return chunk_gen()
        return "Anthropic standard"
    Messages.create = mock_anthropic_create

    def mock_anthropic_stream(self, *args, **kwargs):
        def chunk_gen():
            yield MockAnthropicMessageChunk("Anthropic ")
            yield MockAnthropicMessageChunk("stream!")
        class MockMessageStream:
            def __enter__(self):
                return chunk_gen()
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockMessageStream()
    Messages.stream = mock_anthropic_stream

    # 3. Patch LangChain
    from langchain_groq import ChatGroq
    _originals["groq_stream"] = getattr(ChatGroq, "stream", None)
    _originals["groq_astream"] = getattr(ChatGroq, "astream", None)

    def mock_groq_stream(self, prompt, **kwargs):
        yield MockLangChainChunk("LangChain ")
        yield MockLangChainChunk("stream ")
        yield MockLangChainChunk("chunk!")
    ChatGroq.stream = mock_groq_stream

    async def mock_groq_astream(self, prompt, **kwargs):
        yield MockLangChainChunk("LangChain ")
        yield MockLangChainChunk("async ")
        yield MockLangChainChunk("chunk!")
    ChatGroq.astream = mock_groq_astream


def teardown_module():
    # Restore original methods
    if "openai_create" in _originals:
        from openai.resources.chat.completions import Completions
        Completions.create = _originals["openai_create"]
        
    if "anthropic_create" in _originals:
        from anthropic.resources.messages import Messages
        Messages.create = _originals["anthropic_create"]
        if hasattr(Messages, "stream"):
            delattr(Messages, "stream")

    if "groq_stream" in _originals:
        from langchain_groq import ChatGroq
        if _originals["groq_stream"]:
            ChatGroq.stream = _originals["groq_stream"]
        else:
            delattr(ChatGroq, "stream")
        if _originals["groq_astream"]:
            ChatGroq.astream = _originals["groq_astream"]
        else:
            delattr(ChatGroq, "astream")


from sequa.cassette import cassette
from sequa import storage

def test_langchain_sync_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        from langchain_groq import ChatGroq
        model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")

        # 1. Record Mode
        with cassette(tmpdir, mode="record"):
            stream = model.stream("Test prompt")
            chunks = list(stream)
            assert "".join(c.content for c in chunks) == "LangChain stream chunk!"

        # 2. Replay Mode
        with cassette(tmpdir, mode="replay"):
            stream = model.stream("Test prompt")
            chunks_replayed = list(stream)
            assert "".join(c.content for c in chunks_replayed) == "LangChain stream chunk!"
            assert all(isinstance(c, MockLangChainChunk) for c in chunks_replayed)

@pytest.mark.anyio
async def test_langchain_async_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        from langchain_groq import ChatGroq
        model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")

        # 1. Record Mode
        with cassette(tmpdir, mode="record"):
            stream = model.astream("Test prompt")
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert "".join(c.content for c in chunks) == "LangChain async chunk!"

        # 2. Replay Mode
        with cassette(tmpdir, mode="replay"):
            stream = model.astream("Test prompt")
            chunks_replayed = []
            async for chunk in stream:
                chunks_replayed.append(chunk)
            assert "".join(c.content for c in chunks_replayed) == "LangChain async chunk!"
            assert all(isinstance(c, MockLangChainChunk) for c in chunks_replayed)

def test_openai_sync_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        from openai.resources.chat.completions import Completions
        completions = Completions()

        from sequa.llm.adapters.chat import OpenAIAdapter
        adapter = OpenAIAdapter()

        with cassette(tmpdir, mode="record", adapter=adapter):
            stream = completions.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}], stream=True)
            chunks = list(stream)
            assert "".join(c.choices[0].delta.content for c in chunks) == "Hello there!"

        # Replay
        with cassette(tmpdir, mode="replay", adapter=adapter):
            stream = completions.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}], stream=True)
            chunks_replayed = list(stream)
            assert "".join(c.choices[0].delta.content for c in chunks_replayed) == "Hello there!"
            assert all(isinstance(c, MockChatCompletionChunk) for c in chunks_replayed)

def test_anthropic_sync_stream_context_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        from anthropic.resources.messages import Messages
        messages = Messages()

        from sequa.llm.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()

        with cassette(tmpdir, mode="record", adapter=adapter):
            with messages.stream(model="claude-3", messages=[{"role": "user", "content": "hi"}]) as stream:
                chunks = list(stream)
                assert "".join(c.delta.text for c in chunks) == "Anthropic stream!"

        # Replay
        with cassette(tmpdir, mode="replay", adapter=adapter):
            with messages.stream(model="claude-3", messages=[{"role": "user", "content": "hi"}]) as stream:
                chunks_replayed = list(stream)
                assert "".join(c.delta.text for c in chunks_replayed) == "Anthropic stream!"
                assert all(isinstance(c, MockAnthropicMessageChunk) for c in chunks_replayed)

def test_streaming_pii_masking():
    with tempfile.TemporaryDirectory() as tmpdir:
        from langchain_groq import ChatGroq
        model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")

        # Record with mask_pii=True
        with cassette(tmpdir, mode="record", mask_pii=True):
            stream = model.stream("Contact alice@example.com at 555-555-5555")
            chunks = list(stream)
            
        # Replay should work and cassette file should have masked values
        files = [os.path.join(root, f) for root, _, fs in os.walk(tmpdir) for f in fs if f.endswith(".json") and f != "metadata.json"]
        assert len(files) == 1
        cassette_path = files[0]
        cassette_obj = storage.load(cassette_path)
        
        # Verify prompt masking inside recorded request
        msg = cassette_obj.request["messages"][0]
        msg_content = msg["content"] if isinstance(msg, dict) else msg
        assert msg_content == "Contact [EMAIL] at [PHONE]"
