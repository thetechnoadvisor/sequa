from llmcassette.matcher import normalize, hash_request, match
from llmcassette.llm.adapters.base import CanonicalRequest


def test_normalize_cleans_and_structures():
    req1 = {
        "provider": "openai",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello", "extra_field": "ignore_me"}],
        "temperature": None,
    }
    norm = normalize(req1)
    assert norm["provider"] == "openai"
    assert norm["model"] == "gpt-4"
    assert "temperature" not in norm  # None values removed
    assert norm["messages"][0]["content"] == "hello"
    assert "extra_field" not in norm["messages"][0]  # extra message fields normalized away


def test_hash_request_deterministic_sorting():
    req1 = {
        "provider": "openai",
        "model": "gpt-4",
        "params": {"b": 2, "a": 1},
        "messages": [{"role": "user", "content": "hi"}],
    }
    req2 = {
        "messages": [{"role": "user", "content": "hi"}],
        "params": {"a": 1, "b": 2},
        "provider": "openai",
        "model": "gpt-4",
    }
    # Keys are in different order, but sorted hash should be identical
    h1 = hash_request(req1)
    h2 = hash_request(req2)
    assert h1 == h2


def test_hash_request_ignores_fields():
    req1 = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hi"}],
    }
    req2 = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "hi"}],
    }

    # Without ignoring temperature, they should not match
    assert not match(req1, req2)

    # With ignoring temperature, they should match
    assert match(req1, req2, ignore_fields=["temperature"])
