import os
import tempfile
from llmcassette.models import Cassette
from llmcassette.storage import save, load, exists, delete


def test_cassette_serialization_and_storage():
    cassette = Cassette(
        provider="anthropic",
        hash="test-hash",
        request={"messages": [{"role": "user", "content": "hi"}], "model": "claude-3"},
        response={"output": "hello"},
        metadata={"latency": 1.2},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cassette.json")
        
        assert not exists(path)
        save(cassette, path)
        assert exists(path)

        loaded = load(path)
        assert loaded.id == cassette.id
        assert loaded.provider == "anthropic"
        assert loaded.hash == "test-hash"
        assert loaded.request["model"] == "claude-3"
        assert loaded.response["output"] == "hello"
        assert loaded.metadata["latency"] == 1.2

        delete(path)
        assert not exists(path)
