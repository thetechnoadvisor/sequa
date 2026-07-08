import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from llmcassette.cli.main import main


def test_cli_stats_inspect_clean(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock cassette file
        cassette_data = {
            "id": "test-uuid",
            "provider": "groq",
            "created_at": "2026-07-08T23:00:00Z",
            "request": {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "response": {
                "provider": "groq",
                "output": "hello there!",
                "latency": 450.0,
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
            "metadata": {
                "latency_ms": 450.0
            }
        }
        
        cas_path = os.path.join(tmpdir, "test_cas.json")
        with open(cas_path, "w", encoding="utf-8") as f:
            json.dump(cassette_data, f)

        # 1. Test stats command
        with patch("sys.argv", ["llmcassette", "stats", "--path", tmpdir]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
        
        captured = capsys.readouterr()
        assert "Total Cassettes:      1" in captured.out
        assert "Total Latency Saved:  0.45 seconds" in captured.out

        # 2. Test inspect command
        with patch("sys.argv", ["llmcassette", "inspect", "--path", tmpdir]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "test_cas.json" in captured.out
        assert "llama-3.1-8b-instant" in captured.out

        # 3. Test clean command
        with patch("sys.argv", ["llmcassette", "clean", "--path", tmpdir, "--remove-latency", "--remove-timestamps"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "Successfully formatted/cleaned 1 cassettes." in captured.out

        # Read the file back and verify it was cleaned
        with open(cas_path, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
        
        # Verify created_at is cleared/redacted
        assert cleaned_data["created_at"] == ""
        # Verify latency metadata was removed
        assert "latency_ms" not in cleaned_data.get("metadata", {})
        assert "latency" not in cleaned_data.get("response", {})
