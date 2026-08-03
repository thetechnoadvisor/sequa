import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from sequa.cli.main import main


def test_cli_stats_inspect_clean(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
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
        with patch("sys.argv", ["sequa", "stats", "--path", tmpdir]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
        
        captured = capsys.readouterr()
        assert "Total Cassettes:      1" in captured.out
        assert "Total Latency Saved:  0.45 seconds" in captured.out

        # 2. Test inspect command
        with patch("sys.argv", ["sequa", "inspect", "--path", tmpdir]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "test_cas.json" in captured.out
        assert "llama-3.1-8b-instant" in captured.out

        # 3. Test clean command
        with patch("sys.argv", ["sequa", "clean", "--path", tmpdir, "--remove-latency", "--remove-timestamps"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "Successfully formatted/cleaned 1 cassettes." in captured.out

        # Read the file back and verify it was cleaned
        with open(cas_path, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
        
        assert cleaned_data["created_at"] == ""
        assert "latency_ms" not in cleaned_data.get("metadata", {})
        assert "latency" not in cleaned_data.get("response", {})


def test_cli_diff_text_markdown_html(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        cas1 = {
            "id": "hash-1111",
            "hash": "111111111111",
            "provider": "openai",
            "created_at": "2026-08-01T10:00:00Z",
            "request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Question A"}]},
            "response": {"output": "Answer A"}
        }
        cas2 = {
            "id": "hash-2222",
            "hash": "222222222222",
            "provider": "openai",
            "created_at": "2026-08-02T10:00:00Z",
            "request": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Question B"}]},
            "response": {"output": "Answer B"}
        }

        path1 = os.path.join(tmpdir, "cas1.json")
        path2 = os.path.join(tmpdir, "cas2.json")
        with open(path1, "w") as f:
            json.dump(cas1, f)
        with open(path2, "w") as f:
            json.dump(cas2, f)

        # 1. Text diff
        with patch("sys.argv", ["sequa", "diff", "111111111111", "222222222222", "--path", tmpdir]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "Question A" in captured.out
        assert "Question B" in captured.out

        # 2. Markdown diff
        with patch("sys.argv", ["sequa", "diff", "cas1.json", "cas2.json", "--path", tmpdir, "-f", "markdown"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert "# Sequa Cassette Execution Diff" in captured.out
        assert "| Hash | `111111111111` | `222222222222` |" in captured.out

        # 3. HTML file export output
        out_html = os.path.join(tmpdir, "report.html")
        with patch("sys.argv", ["sequa", "diff", "cas1.json", "cas2.json", "--path", tmpdir, "-f", "html", "-o", out_html]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0

        assert os.path.exists(out_html)
        with open(out_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        assert "<!DOCTYPE html>" in html_content
        assert "Sequa Cassette Execution Diff" in html_content
        assert "111111111111" in html_content
        assert "222222222222" in html_content
