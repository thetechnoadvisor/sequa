import os
import pytest
from sequa import cassette, PostgresStorage
from sequa.models import Cassette
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

DB_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "postgresql://postgres:password@localhost:5432/obs_db"


def is_postgres_available() -> bool:
    try:
        import psycopg
        conn = psycopg.connect(DB_URL, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not is_postgres_available(), reason="PostgreSQL docker container not accessible.")
def test_postgres_storage_crud():
    storage = PostgresStorage(db_url=DB_URL, table_name="test_sequa_cassettes_crud")
    storage.clear()

    cas = Cassette(
        provider="openai",
        hash="test-hash-pg-1",
        request={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello pg"}]},
        response={"output": "hello from pg"},
        metadata={"latency": 0.5},
    )

    assert not storage.exists("test-hash-pg-1")
    storage.save(cas, cassette_id="test-hash-pg-1")
    assert storage.exists("test-hash-pg-1")

    loaded = storage.load("test-hash-pg-1")
    assert loaded.provider == "openai"
    assert loaded.response["output"] == "hello from pg"
    assert loaded.metadata["latency"] == 0.5

    keys = storage.list()
    assert "test-hash-pg-1" in keys

    storage.delete("test-hash-pg-1")
    assert not storage.exists("test-hash-pg-1")
    storage.clear()


@pytest.mark.skipif(not is_postgres_available(), reason="PostgreSQL docker container not accessible.")
def test_postgres_cassette_record_and_replay():
    os.environ["DATABASE_URL"] = DB_URL
    storage = PostgresStorage(db_url=DB_URL, table_name="test_sequa_cassettes_pytest")
    storage.clear()
    
    model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="gsk_dummy")
    test_path = "tests/pg_cassette_test"

    # Record mode
    with patch.object(ChatGroq, "invoke", return_value=AIMessage(content="Response stored in Docker Postgres!")):
        with cassette(test_path, mode="record", storage=storage):
            res = model.invoke("Test query for Postgres")
            assert res.content == "Response stored in Docker Postgres!"

    # Verify directly in DB
    import psycopg
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM test_sequa_cassettes_pytest")
            count = cur.fetchone()[0]
            assert count >= 1

    # Replay mode
    def raise_live_error(*args, **kwargs):
        raise RuntimeError("Live call executed during replay!")

    with patch.object(ChatGroq, "invoke", side_effect=raise_live_error):
        with cassette(test_path, mode="replay", storage=storage):
            replayed = model.invoke("Test query for Postgres")
            assert replayed.content == "Response stored in Docker Postgres!"

    # Clean up test table only
    storage.clear()

