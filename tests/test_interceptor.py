from sequa.llm.adapters import RequestInterceptor


def test_interceptor_wraps_real_call():
    def fake_sdk_call(*args, **kwargs):
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    interceptor = RequestInterceptor()
    res = interceptor.intercept(
        fake_sdk_call,
        {"messages": [{"role": "user", "content": "hello"}]},
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )
    canonical_request = res["request"]
    canonical_response = res["response"]
    result = res["raw"]

    assert canonical_request.provider == "langchain_groq"
    assert canonical_request.messages[0]["content"] == "hello"
    assert canonical_response.output == "hi"
    assert result["choices"][0]["message"]["content"] == "hi"
