from sequa.cassette import cassette
from sequa.llm.adapters import AdapterRegistry, CanonicalRequest, CanonicalResponse, OpenAIAdapter, LangChainGroqAdapter, LangChainGroqMonkeyPatch

def test_openai_adapter_maps_request_and_response():
    adapter = OpenAIAdapter()
    request = adapter.to_canonical_request(
        request={"messages": [{"role": "user", "content": "hello"}]},
        model="gpt-4o-mini",
        temperature=0.2,
    )

    assert request.provider == "openai"
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

    res = fake_llm_call(
        {"messages": [{"role": "user", "content": "hello"}]},
        model="llama-3.1-8b-instant",
    )
    canonical_request = res["request"]
    canonical_response = res["response"]
    result = res["raw"]

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
    assert result["raw"]["choices"][0]["message"]["content"] == "hello"


def test_double_patching_does_not_nest():
    class DummyModel:
        def invoke(self, value, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": value}}]}

    monkey_patch = LangChainGroqMonkeyPatch(adapter=LangChainGroqAdapter())
    patched = monkey_patch.patch(DummyModel)
    patched = monkey_patch.patch(DummyModel) # patch a second time

    result = patched().invoke("hello")
    # If double-patched, result["raw"] would be the dictionary returned from the first patch.
    # If not double-patched, result["raw"] is the raw output from DummyModel.invoke.
    assert isinstance(result["raw"], dict)
    assert "choices" in result["raw"]
    assert "raw" not in result["raw"]
