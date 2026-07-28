from __future__ import annotations

import tempfile
import pytest
from sequa.cassette import cassette
from sequa.llm.adapters import OpenAIAdapter, AnthropicAdapter, LangChainGroqAdapter
from sequa.matcher import hash_request


def test_openai_adapter_tool_calls_request_and_response():
    adapter = OpenAIAdapter()

    # Request with tools and tool message history
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": "18C"}'},
    ]

    canonical_req = adapter.to_canonical_request(
        request={"messages": messages, "tools": tools, "tool_choice": "auto"},
        model="gpt-4o",
    )

    assert canonical_req.provider == "openai"
    assert canonical_req.model == "gpt-4o"
    assert "tools" in canonical_req.params
    assert canonical_req.messages[1]["tool_calls"][0]["id"] == "call_123"
    assert canonical_req.messages[2]["tool_call_id"] == "call_123"

    # Response returning tool calls
    response_dict = {
        "id": "chatcmpl-tool-1",
        "model": "gpt-4o",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }

    canonical_resp = adapter.to_canonical_response(response_dict, model="gpt-4o")

    assert len(canonical_resp.tool_calls) == 1
    assert canonical_resp.tool_calls[0]["id"] == "call_abc"
    assert canonical_resp.tool_calls[0]["function"]["name"] == "get_weather"

    # Replay response
    replayed = adapter.from_canonical_response(canonical_resp, request={"messages": messages})
    assert hasattr(replayed, "choices")
    choice_msg = replayed.choices[0].message
    assert choice_msg.tool_calls is not None
    assert choice_msg.tool_calls[0].id == "call_abc"


def test_anthropic_adapter_tool_use_request_and_response():
    adapter = AnthropicAdapter()

    messages = [
        {"role": "user", "content": "Check weather"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking..."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny"}
            ],
        },
    ]

    canonical_req = adapter.to_canonical_request(
        request={}, messages=messages, model="claude-3-5-sonnet-20241022", tools=[]
    )

    assert canonical_req.provider == "anthropic"
    assert canonical_req.messages[1]["content"][1]["type"] == "tool_use"

    response_dict = {
        "id": "msg_tool_1",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "I will call the tool."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"location": "Paris"},
            },
        ],
        "usage": {"input_tokens": 15, "output_tokens": 25},
    }

    canonical_resp = adapter.to_canonical_response(response_dict)

    assert canonical_resp.output == "I will call the tool."
    assert len(canonical_resp.tool_calls) == 1
    assert canonical_resp.tool_calls[0]["id"] == "toolu_01"
    assert canonical_resp.tool_calls[0]["name"] == "get_weather"

    replayed = adapter.from_canonical_response(canonical_resp, request={})
    assert replayed.stop_reason == "tool_use"
    assert len(replayed.content) == 2
    assert getattr(replayed.content[1], "type") == "tool_use"
    assert getattr(replayed.content[1], "name") == "get_weather"


def test_langchain_groq_adapter_tool_calls():
    adapter = LangChainGroqAdapter()

    class FakeAIMessage:
        def __init__(self):
            self.content = ""
            self.tool_calls = [
                {"name": "search", "args": {"q": "python"}, "id": "call_1", "type": "tool_call"}
            ]
            self.invalid_tool_calls = []
            self.response_metadata = {"model_name": "llama-3.1-8b-instant"}
            self.id = "lc_msg_1"

    canonical_resp = adapter.to_canonical_response(FakeAIMessage())

    assert len(canonical_resp.tool_calls) == 1
    assert canonical_resp.tool_calls[0]["name"] == "search"
    assert canonical_resp.tool_calls[0]["args"] == {"q": "python"}

    replayed_dict = adapter.from_canonical_response(
        canonical_resp, request={"messages": []}
    )
    assert replayed_dict["choices"][0]["message"]["tool_calls"][0]["name"] == "search"


def test_cassette_record_and_replay_with_tool_calls():
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter = OpenAIAdapter()

        def live_llm_call(payload, **kwargs):
            return {
                "id": "chatcmpl-test-tool",
                "model": "gpt-4o",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expr": "2+2"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        request_payload = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [{"type": "function", "function": {"name": "calculate"}}],
        }

        # 1. Record mode
        with cassette(path=tmp_dir, mode="record", adapter=adapter) as cas:
            res_record = cas.engine.handle_call(
                adapter, live_llm_call, request_payload, model="gpt-4o"
            )
            tc_record = res_record["choices"][0]["message"]["tool_calls"]
            assert tc_record[0]["function"]["name"] == "calculate"

        # 2. Replay mode
        with cassette(path=tmp_dir, mode="replay", adapter=adapter) as cas_replay:
            def fail_call(*args, **kwargs):
                raise RuntimeError("Should not be called in replay mode!")

            res_replay = cas_replay.engine.handle_call(
                adapter, fail_call, request_payload, model="gpt-4o"
            )
            tc_replay = res_replay.choices[0].message.tool_calls
            assert tc_replay is not None
            assert tc_replay[0].id == "call_xyz"
            assert tc_replay[0].function.name == "calculate"


def test_hash_request_differentiates_tool_calls():
    adapter = OpenAIAdapter()

    req1 = adapter.to_canonical_request(
        request={
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f1"}}
                    ],
                },
            ]
        },
        model="gpt-4o",
    )

    req2 = adapter.to_canonical_request(
        request={
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c2", "type": "function", "function": {"name": "f2"}}
                    ],
                },
            ]
        },
        model="gpt-4o",
    )

    hash1 = hash_request(req1)
    hash2 = hash_request(req2)
    assert hash1 != hash2
