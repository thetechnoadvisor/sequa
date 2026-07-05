from llmcassette.cassette import cassette
from llmcassette.llm.adapters import AdapterRegistry, CanonicalRequest, CanonicalResponse, OpenAIAdapter, LangChainGroqAdapter, LangChainGroqMonkeyPatch

def test_openai_adapter_maps_request_and_response():
    adapter = OpenAIAdapter()
    request = adapter.to_canonical_request(
        request={"messages": [{"role": "user", "content": "hello"}]},
        model="gpt-4o-mini",
        temperature=0.2,
    )

    assert request.provider == "openai"
    assert request.operation == "chat.completions.create"
    assert request.model == "gpt-4o-mini"
    assert request.messages[0]["content"] == "hello"
    assert request.params["temperature"] == 0.2

    response = adapter.to_canonical_response(
        {
            "id": "resp-1",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        model="gpt-4o-mini",
    )

    assert response.output is not None
    assert response.usage["prompt_tokens"] == 1


def test_registry_registers_and_resolves_adapters():
    registry = AdapterRegistry()
    registry.register(OpenAIAdapter())
    registry.register(LangChainGroqAdapter())

    adapter = registry.get("openai")
    assert isinstance(adapter, OpenAIAdapter)

    groq_adapter = registry.get("langchain_groq")
    assert isinstance(groq_adapter, LangChainGroqAdapter)


def test_langchain_groq_adapter_maps_request_and_response():
    adapter = LangChainGroqAdapter()
    request = adapter.to_canonical_request(
        request={"messages": [{"role": "user", "content": "hello"}]},
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )

    assert request.provider == "langchain_groq"
    assert request.operation == "chat.completions.create"
    assert request.model == "llama-3.1-8b-instant"
    assert request.params["temperature"] == 0.1

    response = adapter.to_canonical_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
        model="llama-3.1-8b-instant",
    )

    assert response.output == "hi"
    assert response.usage["prompt_tokens"] == 2


def test_cassette_decorator_can_wrap_callable():
    @cassette(adapter=LangChainGroqAdapter())
    def fake_llm_call(*args, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    canonical_request, canonical_response, result = fake_llm_call(
        {"messages": [{"role": "user", "content": "hello"}]},
        model="llama-3.1-8b-instant",
    )

    assert canonical_request.provider == "langchain_groq"
    assert canonical_response.output == "hi"
    assert result["choices"][0]["message"]["content"] == "hi"


def test_monkey_patch_wraps_invoke_method():
    class DummyModel:
        def invoke(self, value, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": value}}]}

    monkey_patch = LangChainGroqMonkeyPatch(adapter=LangChainGroqAdapter())
    patched = monkey_patch.patch(DummyModel)

    result = patched().invoke("hello")
    assert result["choices"][0]["message"]["content"] == "hello"
