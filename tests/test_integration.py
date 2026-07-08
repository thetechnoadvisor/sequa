import os
import tempfile
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from llmcassette.cassette import cassette
from llmcassette.recorder import CassetteNotFoundError


def test_groq_record_and_replay_flow():
    # Setup mock original invoke method
    original_invoke = ChatGroq.invoke
    mock_live_call = MagicMock(
        return_value=AIMessage(
            content="Hello from mock live Groq!",
            response_metadata={"model_name": "llama-3.1-8b-instant", "total_time": 0.5},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            id="msg-123",
        )
    )
    ChatGroq.invoke = mock_live_call

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Record Mode
            # Under a record cassette, it should make a live call and save the cassette
            with cassette(tmpdir, mode="record") as cas:
                model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")
                res = model.invoke("say hello")

                assert res.content == "Hello from mock live Groq!"
                assert res.id == "msg-123"
                assert mock_live_call.call_count == 1

            # Check file was saved. We can search files in tmpdir.
            files = os.listdir(tmpdir)
            assert len(files) == 1
            cassette_file_path = os.path.join(tmpdir, files[0])

            # Reset call count
            mock_live_call.reset_mock()
            # If we make a live call now, it should fail (so we ensure replay actually doesn't call it)
            mock_live_call.side_effect = Exception("Live call should not be made!")

            # 2. Replay Mode
            # It should load the cached response and NOT call the live method
            with cassette(tmpdir, mode="replay"):
                res2 = model.invoke("say hello")
                assert res2.content == "Hello from mock live Groq!"
                assert res2.id == "msg-123"
                assert res2.response_metadata["model_name"] == "llama-3.1-8b-instant"
                assert mock_live_call.call_count == 0

            # 3. Auto Mode (Cache Hit)
            # Replays the cassette directly
            with cassette(tmpdir, mode="auto"):
                res3 = model.invoke("say hello")
                assert res3.content == "Hello from mock live Groq!"
                assert mock_live_call.call_count == 0

            # 4. Replay Mode - Missing Cassette Exception
            # Calling with a different prompt should raise CassetteNotFoundError
            with pytest.raises(CassetteNotFoundError):
                with cassette(tmpdir, mode="replay"):
                    model.invoke("a completely different prompt")

    finally:
        # Restore ChatGroq.invoke
        ChatGroq.invoke = original_invoke


def test_ignore_fields_matching():
    original_invoke = ChatGroq.invoke
    mock_live_call = MagicMock(
        return_value=AIMessage(
            content="Answer",
            response_metadata={"model_name": "llama-3.1-8b-instant"},
            id="msg-1",
        )
    )
    ChatGroq.invoke = mock_live_call

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record with temperature 0.1
            with cassette(tmpdir, mode="record", ignore_fields=["temperature"]):
                model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake", temperature=0.1)
                model.invoke("say hello")
            
            mock_live_call.reset_mock()
            mock_live_call.side_effect = Exception("Should have replayed!")

            # Replay with temperature 0.9. Since we ignore temperature, it should match!
            with cassette(tmpdir, mode="replay", ignore_fields=["temperature"]):
                model2 = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake", temperature=0.9)
                res = model2.invoke("say hello")
                assert res.content == "Answer"
                assert mock_live_call.call_count == 0

    finally:
        ChatGroq.invoke = original_invoke


def test_custom_normalizer():
    original_invoke = ChatGroq.invoke
    mock_live_call = MagicMock(
        return_value=AIMessage(
            content="Normalized Output",
            response_metadata={"model_name": "llama-3.1-8b-instant"},
            id="msg-norm",
        )
    )
    ChatGroq.invoke = mock_live_call

    def my_normalizer(req_dict):
        # Redact the model name or change it to a standard value
        req_dict["model"] = "standard-model"
        return req_dict

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record with model modelA
            with cassette(tmpdir, mode="record", normalizer=my_normalizer):
                modelA = ChatGroq(model_name="modelA", groq_api_key="fake")
                modelA.invoke("say hello")
            
            mock_live_call.reset_mock()
            mock_live_call.side_effect = Exception("Should have replayed due to normalizer!")

            # Replay with model modelB. Since normalizer changes it to standard-model, they will match!
            with cassette(tmpdir, mode="replay", normalizer=my_normalizer):
                modelB = ChatGroq(model_name="modelB", groq_api_key="fake")
                res = modelB.invoke("say hello")
                assert res.content == "Normalized Output"
                assert mock_live_call.call_count == 0

    finally:
        ChatGroq.invoke = original_invoke
