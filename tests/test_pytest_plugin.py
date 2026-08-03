import os
import tempfile
from unittest.mock import MagicMock
import pytest

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from sequa.cassette import cassette

pytest_plugins = ["pytester"]


def test_fixture_sequa_cassette(request):
    original_invoke = ChatGroq.invoke
    mock_live_call = MagicMock(
        return_value=AIMessage(
            content="Hello from fixture!",
            response_metadata={"model_name": "llama-3.1-8b-instant"},
            id="msg-fixture",
        )
    )
    ChatGroq.invoke = mock_live_call

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fixture_test.json")

            # First run: record
            with cassette(path=path, mode="record"):
                model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")
                res = model.invoke("test fixture")
                assert res.content == "Hello from fixture!"
                assert mock_live_call.call_count == 1

            assert os.path.exists(path)

            mock_live_call.reset_mock()
            mock_live_call.side_effect = Exception("Should replay from cassette!")

            # Second run: replay
            with cassette(path=path, mode="replay"):
                res2 = model.invoke("test fixture")
                assert res2.content == "Hello from fixture!"
                assert mock_live_call.call_count == 0

    finally:
        ChatGroq.invoke = original_invoke


def test_pytester_marker_and_fixture_integration(pytester):
    """Test full pytest execution using @pytest.mark.sequa and sequa_cassette fixture."""
    pytester.makepyfile(
        """
        import pytest
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage
        from langchain_groq import ChatGroq

        def setup_mock(monkeypatch):
            mock_live = MagicMock(
                return_value=AIMessage(
                    content="Plugin Success!",
                    response_metadata={"model_name": "llama-3.1-8b-instant"},
                    id="msg-plugin"
                )
            )
            monkeypatch.setattr(ChatGroq, "invoke", mock_live)
            return mock_live

        @pytest.mark.sequa(mode="record")
        def test_record_flow(monkeypatch, sequa_cassette):
            mock = setup_mock(monkeypatch)
            model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")
            res = model.invoke("plugin test")
            assert res.content == "Plugin Success!"
            assert mock.call_count == 1
            assert sequa_cassette is not None

        @pytest.mark.cassette(mode="record")
        def test_alias_marker_flow(monkeypatch):
            mock = setup_mock(monkeypatch)
            model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")
            res = model.invoke("alias marker test")
            assert res.content == "Plugin Success!"
            assert mock.call_count == 1
        """
    )

    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=2)


def test_pytester_cli_mode_override(pytester):
    """Test --sequa-mode override via pytest CLI."""
    pytester.makepyfile(
        """
        import pytest
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage
        from langchain_groq import ChatGroq

        @pytest.mark.sequa(mode="record")
        def test_cli_override(monkeypatch):
            mock_live = MagicMock(
                return_value=AIMessage(
                    content="CLI Override",
                    response_metadata={"model_name": "llama-3.1-8b-instant"}
                )
            )
            monkeypatch.setattr(ChatGroq, "invoke", mock_live)
            model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="fake")
            res = model.invoke("cli test")
            assert res.content == "CLI Override"
        """
    )

    result = pytester.runpytest("--sequa-mode=record", "-v")
    result.assert_outcomes(passed=1)


def test_pytester_disable_sequa(pytester):
    """Test --disable-sequa disables Sequa interception."""
    pytester.makepyfile(
        """
        import pytest
        from sequa.cassette import get_active_engine

        @pytest.mark.sequa
        def test_disabled(sequa_cassette):
            assert sequa_cassette is None
            assert get_active_engine() is None
        """
    )

    result = pytester.runpytest("--disable-sequa", "-v")
    result.assert_outcomes(passed=1)
