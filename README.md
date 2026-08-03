# Sequa 📼

> **Deterministic testing for AI applications.**
>
> **Record once. Replay forever.**

[![PyPI](https://img.shields.io/pypi/v/sequa)](https://pypi.org/project/sequa/)
[![Python](https://img.shields.io/pypi/pyversions/sequa)](https://pypi.org/project/sequa/)
[![License](https://img.shields.io/github/license/thetechnoadvisor/sequa)](LICENSE)

---

## Stop paying for every AI test run.

Every time your AI application runs during testing, it probably:

- 💸 Calls the LLM again
- 🐢 Slows down your CI pipeline
- 🎲 Produces slightly different outputs
- 🌐 Depends on internet connectivity

**Sequa records a real AI execution once and replays it locally during future test runs.**

### The result

- ⚡ Millisecond replay
- 💰 Zero replay API costs
- 🧪 Deterministic testing
- 💻 Works offline

---

# Before

```python
from langchain_groq import ChatGroq

model = ChatGroq(model_name="llama-3.1-8b-instant")

response = model.invoke(
    "Write a 3-word slogan for gravity."
)

# ⏱️ 2.3 seconds
# 🌐 Live API Call
```

---

# After

```python
from langchain_groq import ChatGroq
from sequa import cassette

model = ChatGroq(model_name="llama-3.1-8b-instant")

with cassette("tests/cassettes"):
    response = model.invoke(
        "Write a 3-word slogan for gravity."
    )

# First Run
# ⏱️ 2.3 seconds
# 🌐 Live API Call
# 💾 Recorded

# Every Run After
# ⏱️ 12 ms
# ❌ No API Calls
# 📼 Replayed Locally
```

---

# Why Sequa?

| Without Sequa | With Sequa |
|---------------|------------|
| Calls the LLM on every test | Record once, replay forever |
| Seconds of latency | Millisecond replay |
| API cost every execution | No replay API cost |
| Internet required | Works offline |
| Non-deterministic | Deterministic |

---

# Supported Frameworks

- ✅ OpenAI
- ✅ Anthropic
- ✅ LangChain
- ✅ LangGraph

---

# Installation

```bash
pip install sequa
```

or

```bash
uv add sequa
```

---

# Quick Start

```python
from langchain_groq import ChatGroq
from sequa import cassette

model = ChatGroq(model_name="llama-3.1-8b-instant")

with cassette("tests/cassettes"):
    response = model.invoke("Hello Sequa!")
```

That's it.

The first execution records the response.

Every matching execution after that replays it locally without calling the LLM.

---

# Features

- 🔎 **Searchable AI Executions & Instant Replay (v0.5.0)**: Find past executions with TF-IDF cosine similarity (`sequa search "refund" --since yesterday`) and generate local replay code (`sequa replay <hash>`).
- 📼 Record once, replay forever
- ⚡ Replay, Record, Auto and Live execution modes
- 🧰 Tool Calling & Function Calling support
- 🌊 Streaming support (sync & async)
- 🔒 PII & Sensitive Information Masking
- 🛡️ NVIDIA NeMo Guardrails Integration
- 🧠 Deterministic request hashing
- 🎯 Custom ignored fields
- 🔧 Custom request normalizers
- 🗂️ File, Memory & PostgreSQL storage backends
- 🧹 CLI utilities (`search`, `replay`, `log`, `stats`, `inspect`, `clean`)

---

# Searchable AI Executions & Instant Replay (v0.5.0)

When a bug is reported (*"The AI gave the wrong answer yesterday"*), Sequa makes it reproducible in seconds:

```bash
# 1. Search past executions by TF-IDF cosine similarity & relative time
sequa search "refund request failed" --since yesterday -i

# 2. Inspect target execution and get instant Python replay code
sequa replay 1cea06570793

# 3. View chronological Git-like execution history
sequa log --path cassettes -n 5
```

Or query programmatically via Python API:

```python
from sequa import search_cassettes

# Search recorded executions by natural language & metadata
results = search_cassettes(query="refund request", since="yesterday", provider="openai")

for res in results:
    print(f"Match Score: {res.score:.4f} | Hash: {res.hash[:12]}")
    print(f"  Input:  {res.input_snippet}")
    print(f"  Output: {res.output_snippet}")
```

---

# CLI Reference

Sequa includes a built-in CLI to search, inspect, format, and debug your recorded cassettes:

| Command | Description | Example |
| :--- | :--- | :--- |
| `sequa diff` | Compare two cassette executions (text, markdown, html formats) | `sequa diff hash1 hash2 -f html -o diff.html` |
| `sequa search` | Search executions by vector cosine similarity, time window, or provider/model | `sequa search "refund" --since yesterday -i` |
| `sequa log` | Show Git-like chronological execution log | `sequa log --path cassettes -n 10` |
| `sequa replay` | Inspect a target cassette and generate copy-paste Python replay code | `sequa replay 1cea06570793` |
| `sequa stats` | View total cassette count, disk size, and saved API latency | `sequa stats --path cassettes` |
| `sequa inspect` | List all saved cassettes with providers, models, and timestamps | `sequa inspect --path cassettes` |
| `sequa clean` | Redact volatile timestamps and latency before committing to Git | `sequa clean --remove-latency --remove-timestamps` |

---

# Cassette Execution Diffing (`sequa diff`) 🔍

Compare any two recorded cassettes by hash, ID, or file path to inspect prompt, model, parameter, or output differences:

```bash
# 1. Compare two cassette executions in terminal text mode
sequa diff 1cea06570793 4b93d6e3 --path cassettes

# 2. Export diff comparison report to GitHub Markdown
sequa diff 1cea06570793 4b93d6e3 -f markdown -o diff.md

# 3. Export standalone styled HTML diff report
sequa diff 1cea06570793 4b93d6e3 -f html -o diff.html

# 4. Interactive Diff from Search Results
sequa search "refund request" -i
# Enter result numbers to diff (e.g., '1,2' or 'diff 1 2'):
```

---

# Pytest Integration 🧪

Sequa includes built-in Pytest support via the `sequa` plugin.

### Markers & Fixtures

Use `@pytest.mark.sequa` (or alias `@pytest.mark.cassette`) or inject the `sequa_cassette` fixture:

```python
import pytest
from langchain_groq import ChatGroq

@pytest.mark.sequa(mode="auto")
def test_llm_feature():
    model = ChatGroq(model_name="llama-3.1-8b-instant")
    response = model.invoke("Say hello")
    assert "hello" in response.content.lower()

def test_with_fixture(sequa_cassette):
    model = ChatGroq(model_name="llama-3.1-8b-instant")
    response = model.invoke("Hello world")
```

### Pytest CLI Flags

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--sequa-mode=<mode>` | Globally override cassette mode (`auto`, `record`, `replay`, `live`) | Marker / `auto` |
| `--sequa-path=<path>` | Set base cassette directory | `tests/cassettes` |
| `--sequa-mask-pii` | Enable PII masking across all test cassette recordings | `False` |
---

# GitHub Actions Integration 🐙

Run deterministic LLM snapshot tests in CI/CD with zero API costs using the official Sequa GitHub Action.

### Quick Workflow Setup

Add `.github/workflows/ai-tests.yml` to your repository:

```yaml
name: AI Snapshot Tests

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Sequa LLM Tests
        uses: thetechnoadvisor/sequa@v1
        with:
          mode: replay
          cassette-path: tests/cassettes
          post-summary: true
```

### Action Options

| Input | Description | Default |
| :--- | :--- | :--- |
| `mode` | Execution mode (`replay`, `auto`, `record`, `live`) | `replay` |
| `cassette-path` | Path to saved cassette files directory | `tests/cassettes` |
| `python-version` | Python version for setup | `3.12` |
| `pytest-args` | Additional arguments passed to pytest | `""` |
| `post-summary` | Post cassette stats report to `$GITHUB_STEP_SUMMARY` | `true` |

---


# Common Use Cases

### 🚀 Speed up AI integration tests

Run your test suite in milliseconds instead of waiting for repeated LLM calls.

---

### 💰 Reduce API costs

Replay previously recorded executions without paying for another API request.

---

### 🧪 Deterministic testing

Replay the exact same execution every time.

---

### 💻 Offline development

Develop and test AI applications without internet connectivity.

---

### 🐞 Reproduce bugs

Replay the exact LLM interaction that caused the issue.

---

# Storage Backends

Sequa supports multiple storage backends.

- 📁 File Storage
- 🧠 In-Memory Storage
- 🐘 PostgreSQL Storage

Choose whichever fits your workflow.

---

# Documentation

Comprehensive documentation is available at:

👉 https://sequa.thetechnoadvisor.com/docs

---

# Contributing

Contributions are always welcome.

- ⭐ Star the repository
- 🐞 Report bugs
- 💡 Suggest new features
- 🔧 Open a Pull Request

---

# License

MIT License.