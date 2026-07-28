import os
import pytest
from sequa import Cassette, FileStorage, MemoryStorage, StorageBackend, cassette


def test_storage_backend_abc():
    with pytest.raises(TypeError):
        StorageBackend()  # type: ignore


def test_memory_storage_crud():
    storage = MemoryStorage()
    assert storage.list() == []

    cas = Cassette(
        provider="test-provider",
        hash="hash123",
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"output": "hello"},
    )

    # Save to memory storage
    storage.save(cas, cassette_id="cas_1")
    assert storage.exists("cas_1")
    assert storage.exists("hash123")
    assert storage.list() == ["cas_1"]

    # Load from memory storage
    loaded = storage.load("cas_1")
    assert loaded.id == cas.id
    assert loaded.provider == "test-provider"
    assert loaded.response["output"] == "hello"

    # Delete from memory storage
    storage.delete("cas_1")
    assert not storage.exists("cas_1")
    assert storage.list() == []


def test_cassette_model_with_memory_storage():
    mem_storage = MemoryStorage()
    c = Cassette(
        provider="openai",
        hash="test-hash",
        request={"model": "gpt-4"},
        response={"output": "test output"},
        storage=mem_storage,
    )

    # Save via cassette.save()
    c.save(path_or_id="custom_id")
    assert mem_storage.exists("custom_id")

    loaded = Cassette.load("custom_id", storage=mem_storage)
    assert loaded.request["model"] == "gpt-4"
    assert loaded.response["output"] == "test output"


def test_no_disk_files_created_with_memory_storage():
    mem_storage = MemoryStorage()
    dummy_dir = "test_memory_storage_dummy_dir"
    assert not os.path.exists(dummy_dir)

    with cassette(path=dummy_dir, storage=mem_storage, mode="auto") as cas_cm:
        cas_cm.engine.storage.save(
            Cassette(
                provider="openai",
                hash="test-hash-123",
                request={"messages": [{"role": "user", "content": "Hello"}]},
                response={"output": "World"},
            ),
            cassette_id="test_key",
        )

    assert len(mem_storage.list()) == 1
    assert mem_storage.exists("test_key")
    assert not os.path.exists(dummy_dir)

def test_storage_string_options():
    from sequa.storage import FileStorage, MemoryStorage, resolve_storage

    # Test string resolution
    mem_st = resolve_storage("memory")
    assert isinstance(mem_st, MemoryStorage)

    file_st = resolve_storage("file", base_dir="test_dir")
    assert isinstance(file_st, FileStorage)
    assert file_st.base_dir == "test_dir"

    with pytest.raises(ValueError):
        resolve_storage("invalid_option")

    # Test cassette context manager with string storage option
    dummy_dir = "test_string_option_dummy_dir"
    with cassette(path=dummy_dir, storage="memory", mode="auto") as cas_cm:
        assert isinstance(cas_cm.engine.storage, MemoryStorage)
        cas_cm.engine.storage.save(
            Cassette(
                provider="anthropic",
                hash="string-opt-hash",
                request={"messages": [{"role": "user", "content": "Hello string storage"}]},
                response={"output": "Response from memory"},
            ),
            cassette_id="string_opt_key",
        )
        assert cas_cm.engine.storage.exists("string_opt_key")

    # Ensure no disk directory created
    assert not os.path.exists(dummy_dir)




