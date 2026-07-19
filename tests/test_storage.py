import os
import tempfile
from sequa.models import Cassette
from sequa.storage import save, load, exists, delete


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


def test_provider_folders_and_metadata_index():
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        anthropic_cas = Cassette(
            provider="anthropic",
            hash="7f3a9c1e",
            request={"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "hi"}]},
            response={"output": "hello from claude"},
            metadata={"latency_ms": 120.0},
        )
        openai_cas = Cassette(
            provider="openai",
            hash="91ab2d8c",
            request={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            response={"output": "hello from gpt4o"},
            metadata={"latency_ms": 95.0},
        )

        anthropic_path = os.path.join(tmpdir, "anthropic", "7f3a9c1e.json")
        openai_path = os.path.join(tmpdir, "openai", "91ab2d8c.json")

        save(anthropic_cas, anthropic_path, base_dir=tmpdir)
        save(openai_cas, openai_path, base_dir=tmpdir)

        # Verify folder structure
        assert os.path.exists(os.path.join(tmpdir, "anthropic", "7f3a9c1e.json"))
        assert os.path.exists(os.path.join(tmpdir, "openai", "91ab2d8c.json"))
        assert os.path.exists(os.path.join(tmpdir, "metadata.json"))

        # Verify metadata.json contents
        with open(os.path.join(tmpdir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["total_cassettes"] == 2
        assert metadata["providers"]["anthropic"] == 1
        assert metadata["providers"]["openai"] == 1

        assert "7f3a9c1e" in metadata["cassettes"]
        assert metadata["cassettes"]["7f3a9c1e"]["provider"] == "anthropic"
        assert metadata["cassettes"]["7f3a9c1e"]["file"] == "anthropic/7f3a9c1e.json"

        assert "91ab2d8c" in metadata["cassettes"]
        assert metadata["cassettes"]["91ab2d8c"]["provider"] == "openai"
        assert metadata["cassettes"]["91ab2d8c"]["file"] == "openai/91ab2d8c.json"

