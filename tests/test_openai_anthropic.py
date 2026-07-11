from __future__ import annotations

import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest

# -----------------------------------------------------------------------------
# Dynamic Mock setup for openai and anthropic modules
# -----------------------------------------------------------------------------

# OpenAI mock classes
class MockCompletions:
    def create(self, *args, **kwargs):
        class MockChoiceMessage:
            role = "assistant"
            content = "Hello from mock OpenAI!"

        class MockChoice:
            finish_reason = "stop"
            index = 0
            message = MockChoiceMessage()

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        class MockChatCompletion:
            id = "msg-openai-123"
            model = "gpt-4"
            choices = [MockChoice()]
            usage = MockUsage()

        return MockChatCompletion()


class MockAsyncCompletions:
    async def create(self, *args, **kwargs):
        class MockChoiceMessage:
            role = "assistant"
            content = "Hello from mock async OpenAI!"

        class MockChoice:
            finish_reason = "stop"
            index = 0
            message = MockChoiceMessage()

        class MockChatCompletion:
            id = "msg-openai-async"
            model = "gpt-4"
            choices = [MockChoice()]
            usage = None

        return MockChatCompletion()


# Register OpenAI modules in sys.modules
openai_mod = types.ModuleType("openai")
sys.modules["openai"] = openai_mod

openai_resources_mod = types.ModuleType("openai.resources")
sys.modules["openai.resources"] = openai_resources_mod

openai_chat_mod = types.ModuleType("openai.resources.chat")
sys.modules["openai.resources.chat"] = openai_chat_mod

openai_completions_mod = types.ModuleType("openai.resources.chat.completions")
openai_completions_mod.Completions = MockCompletions
openai_completions_mod.AsyncCompletions = MockAsyncCompletions
sys.modules["openai.resources.chat.completions"] = openai_completions_mod


# Anthropic mock classes
class MockMessages:
    def create(self, *args, **kwargs):
        class MockTextBlock:
            text = "Hello from mock Anthropic!"
            type = "text"

        class MockUsage:
            input_tokens = 5
            output_tokens = 10

        class MockMessage:
            id = "msg-anthropic-123"
            model = "claude-3"
            role = "assistant"
            type = "message"
            content = [MockTextBlock()]
            usage = MockUsage()
            stop_reason = "end_turn"
            stop_sequence = None

        return MockMessage()


class MockAsyncMessages:
    async def create(self, *args, **kwargs):
        class MockTextBlock:
            text = "Hello from mock async Anthropic!"
            type = "text"

        class MockMessage:
            id = "msg-anthropic-async"
            model = "claude-3"
            role = "assistant"
            type = "message"
            content = [MockTextBlock()]
            usage = None
            stop_reason = "end_turn"
            stop_sequence = None

        return MockMessage()


# Register Anthropic modules in sys.modules
anthropic_mod = types.ModuleType("anthropic")
sys.modules["anthropic"] = anthropic_mod

anthropic_resources_mod = types.ModuleType("anthropic.resources")
sys.modules["anthropic.resources"] = anthropic_resources_mod

anthropic_messages_mod = types.ModuleType("anthropic.resources.messages")
anthropic_messages_mod.Messages = MockMessages
anthropic_messages_mod.AsyncMessages = MockAsyncMessages
sys.modules["anthropic.resources.messages"] = anthropic_messages_mod


# Now we can import our cassette library and test
from sequa.cassette import cassette
from sequa.recorder import CassetteNotFoundError


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_openai_record_and_replay_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Record Mode
        with cassette(tmpdir, mode="record", adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()) as cas:
            from openai.resources.chat.completions import Completions
            client = Completions()
            res = client.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "say hello"}],
                temperature=0.7
            )
            assert getattr(res, "id") == "msg-openai-123"
            assert res.choices[0].message.content == "Hello from mock OpenAI!"

        # Check cassette was recorded
        files = os.listdir(tmpdir)
        assert len(files) == 1

        # 2. Replay Mode
        # Override original completions class method to verify it is NOT called during replay
        original_create = MockCompletions.create
        MockCompletions.create = MagicMock(side_effect=Exception("Should have replayed from file!"))

        try:
            with cassette(tmpdir, mode="replay", adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()):
                res2 = client.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "say hello"}],
                    temperature=0.7
                )
                assert getattr(res2, "id") == "msg-openai-123"
                assert res2.choices[0].message.content == "Hello from mock OpenAI!"
                assert res2.usage.total_tokens == 15
        finally:
            MockCompletions.create = original_create


@pytest.mark.anyio
async def test_openai_record_and_replay_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Record Mode
        with cassette(tmpdir, mode="record", adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()):
            from openai.resources.chat.completions import AsyncCompletions
            client = AsyncCompletions()
            res = await client.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "say hello async"}],
            )
            assert getattr(res, "id") == "msg-openai-async"
            assert res.choices[0].message.content == "Hello from mock async OpenAI!"

        # 2. Replay Mode
        original_async_create = MockAsyncCompletions.create
        MockAsyncCompletions.create = MagicMock(side_effect=Exception("Should have replayed from file!"))

        try:
            with cassette(tmpdir, mode="replay", adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()):
                res2 = await client.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "say hello async"}],
                )
                assert getattr(res2, "id") == "msg-openai-async"
                assert res2.choices[0].message.content == "Hello from mock async OpenAI!"
        finally:
            MockAsyncCompletions.create = original_async_create


def test_anthropic_record_and_replay_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Record Mode
        with cassette(tmpdir, mode="record", adapter=sys.modules["sequa.llm.adapters"].AnthropicAdapter()) as cas:
            from anthropic.resources.messages import Messages
            client = Messages()
            res = client.create(
                model="claude-3",
                messages=[{"role": "user", "content": "hi anthropic"}],
            )
            assert getattr(res, "id") == "msg-anthropic-123"
            assert res.content[0].text == "Hello from mock Anthropic!"

        # Check cassette was recorded
        files = os.listdir(tmpdir)
        assert len(files) == 1

        # 2. Replay Mode
        original_create = MockMessages.create
        MockMessages.create = MagicMock(side_effect=Exception("Should have replayed from file!"))

        try:
            with cassette(tmpdir, mode="replay", adapter=sys.modules["sequa.llm.adapters"].AnthropicAdapter()):
                res2 = client.create(
                    model="claude-3",
                    messages=[{"role": "user", "content": "hi anthropic"}],
                )
                assert getattr(res2, "id") == "msg-anthropic-123"
                assert res2.content[0].text == "Hello from mock Anthropic!"
                assert res2.usage.output_tokens == 10
        finally:
            MockMessages.create = original_create


@pytest.mark.anyio
async def test_anthropic_record_and_replay_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Record Mode
        with cassette(tmpdir, mode="record", adapter=sys.modules["sequa.llm.adapters"].AnthropicAdapter()):
            from anthropic.resources.messages import AsyncMessages
            client = AsyncMessages()
            res = await client.create(
                model="claude-3",
                messages=[{"role": "user", "content": "hi async anthropic"}],
            )
            assert getattr(res, "id") == "msg-anthropic-async"
            assert res.content[0].text == "Hello from mock async Anthropic!"

        # 2. Replay Mode
        original_async_create = MockAsyncMessages.create
        MockAsyncMessages.create = MagicMock(side_effect=Exception("Should have replayed from file!"))

        try:
            with cassette(tmpdir, mode="replay", adapter=sys.modules["sequa.llm.adapters"].AnthropicAdapter()):
                res2 = await client.create(
                    model="claude-3",
                    messages=[{"role": "user", "content": "hi async anthropic"}],
                )
                assert getattr(res2, "id") == "msg-anthropic-async"
                assert res2.content[0].text == "Hello from mock async Anthropic!"
        finally:
            MockAsyncMessages.create = original_async_create
