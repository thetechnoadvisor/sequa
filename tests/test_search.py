from __future__ import annotations

import datetime
import os
import tempfile
import pytest

from sequa.models import Cassette
from sequa.search import (
    TFIDFEmbedder,
    cosine_similarity,
    extract_searchable_text,
    parse_created_at,
    parse_time_constraint,
    search_cassettes,
)
from sequa.storage import MemoryStorage


def test_parse_time_constraint():
    now = datetime.datetime.now(datetime.timezone.utc)

    # Relative minutes
    dt10m = parse_time_constraint("10m")
    assert dt10m is not None
    assert (now - dt10m).total_seconds() >= 590

    # Relative hours
    dt2h = parse_time_constraint("2h")
    assert dt2h is not None
    assert (now - dt2h).total_seconds() >= 7100

    # Relative days
    dt1d = parse_time_constraint("1d")
    assert dt1d is not None

    # Shortcuts
    dt_y = parse_time_constraint("yesterday")
    assert dt_y is not None

    # ISO format
    iso_str = "2026-07-31T12:00:00+00:00"
    dt_iso = parse_time_constraint(iso_str)
    assert dt_iso is not None
    assert dt_iso.year == 2026


def test_extract_searchable_text():
    cassette_data = {
        "id": "test-123",
        "provider": "openai",
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful customer support bot."},
                {"role": "user", "content": "How do I request a refund for my subscription?"},
            ]
        },
        "response": {
            "output": "To request a refund, please navigate to Settings > Billing and click Refund."
        },
    }

    full_text, in_snip, out_snip = extract_searchable_text(cassette_data)

    assert "customer support bot" in full_text
    assert "refund for my subscription" in full_text
    assert "Settings > Billing" in full_text
    assert "How do I request a refund" in in_snip
    assert "To request a refund" in out_snip


def test_tfidf_cosine_similarity():
    embedder = TFIDFEmbedder()

    corpus = [
        "How do I cancel my subscription and get a refund?",
        "Write a python script to calculate fibonacci numbers.",
        "What is the capital of France and best places to visit in Paris?",
    ]

    doc_vecs = embedder.fit_transform(corpus)
    q_vec = embedder.transform_query("subscription refund billing")

    sim0 = embedder.dict_cosine_similarity(q_vec, doc_vecs[0])
    sim1 = embedder.dict_cosine_similarity(q_vec, doc_vecs[1])
    sim2 = embedder.dict_cosine_similarity(q_vec, doc_vecs[2])

    assert sim0 > sim1
    assert sim0 > sim2
    assert sim0 > 0.1


def test_search_cassettes_memory():
    storage = MemoryStorage()

    c1 = Cassette(
        id="c1",
        provider="openai",
        created_at="2026-07-31T14:00:00+00:00",
        request={"messages": [{"role": "user", "content": "Explain quantum computing simply."}], "model": "gpt-4o"},
        response={"output": "Quantum computing uses qubits and superposition."},
    )

    c2 = Cassette(
        id="c2",
        provider="anthropic",
        created_at="2026-07-31T14:30:00+00:00",
        request={"messages": [{"role": "user", "content": "Write a fast sorting algorithm in Python."}], "model": "claude-3-5-sonnet"},
        response={"output": "Here is quicksort in Python..."},
    )

    c3 = Cassette(
        id="c3",
        provider="openai",
        created_at="2026-07-31T15:00:00+00:00",
        request={"messages": [{"role": "user", "content": "How to optimize Python quicksort memory usage?"}], "model": "gpt-4o"},
        response={"output": "In-place partitioning reduces auxiliary array overhead."},
    )

    storage.save(c1)
    storage.save(c2)
    storage.save(c3)

    # Search for quicksort algorithm
    results = search_cassettes("quicksort python sorting", storage=storage, top_k=2)
    assert len(results) >= 2
    assert results[0].id in ("c2", "c3")

    # Search with provider filter
    openai_results = search_cassettes("quicksort", provider="openai", storage=storage)
    assert len(openai_results) == 1
    assert openai_results[0].id == "c3"


def test_search_cli_integration(capsys):
    from sequa.cli.main import main
    import sys

    with tempfile.TemporaryDirectory() as tmp_dir:
        c = Cassette(
            id="cli-c1",
            provider="groq",
            created_at="2026-07-31T10:00:00+00:00",
            request={"messages": [{"role": "user", "content": "Hello CLI search!"}], "model": "llama-3"},
            response={"output": "Hello back from llama!"},
        )
        c.save(path_or_id=os.path.join(tmp_dir, "test_cli.json"))

        # Test sequa log
        sys.argv = ["sequa", "log", "-p", tmp_dir]
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

        captured = capsys.readouterr()
        assert "Sequa Cassette Execution Log" in captured.out
        assert "Hello CLI search!" in captured.out

        # Test sequa search
        sys.argv = ["sequa", "search", "search", "-p", tmp_dir]
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

        captured = capsys.readouterr()
        assert "Search Results for:" in captured.out
        assert "Hello CLI search!" in captured.out

        # Test sequa replay
        sys.argv = ["sequa", "replay", "test_cli.json", "-p", tmp_dir]
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

        captured = capsys.readouterr()
        assert "Sequa Replay Target" in captured.out
        assert "with cassette(" in captured.out
