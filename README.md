# LLMCassette 📼

LLMCassette is snapshot testing for LLM applications. Record once, replay forever.

---

## The Magic

### Before
```python
from langchain_groq import ChatGroq

model = ChatGroq(model_name="llama-3.1-8b-instant")
response = model.invoke("Write a 3-word slogan for gravity.")
# ⏱️ Time taken: 2.3 seconds
```

### After
```python
from langchain_groq import ChatGroq
from llmcassette import cassette

model = ChatGroq(model_name="llama-3.1-8b-instant")

with cassette("tests/cassettes"):
    response = model.invoke("Write a 3-word slogan for gravity.")
    # ⏱️ First run: 2.3 seconds (recorded to tests/cassettes/)
    # ⏱️ Second run: 12 ms (replayed locally!)
```

---

## Features

- **Record once, replay forever**: Speed up integration test suites from minutes to milliseconds.
- **Multiple Execution Modes**: Support `replay`, `record`, `auto`, and `live` modes.
- **Robust Key Sorting & Hashing**: Recursively sorts request inputs to generate deterministic hashes.
- **Custom Ignored Fields**: Easily ignore dynamic/unstable fields (e.g. `temperature`, `max_tokens`).
- **Custom Normalizers**: Redact, replace, or clean requests prior to hashing.
- **CLI Utilities**: Inspect, format, and calculate statistics of stored cassettes.

---

## Installation

```bash
uv pip install -e .
```

---

## Configuration & Advanced API

### 1. Execution Modes

Control LLMCassette behavior via the `mode` parameter:

```python
with cassette("tests/cassettes", mode="replay"):
    # Will raise CassetteNotFoundError if no matching cassette is found.
    # Guaranteed to make zero external network requests.
```

- `auto` (Default): Replays if a matching cassette exists, otherwise calls the live API and records it.
- `record`: Always calls the live API and records/overwrites the cassette.
- `replay`: Never calls the live API. Raises `CassetteNotFoundError` on cache misses.
- `live`: Direct pass-through to the live API, bypassing cassettes entirely.

### 2. Ignore Fields

Strip request parameters before generating hashes:

```python
with cassette("tests/cassettes", ignore_fields=["temperature", "max_tokens"]):
    # These two calls generate the exact same hash and match the same cassette:
    model.invoke("hello", temperature=0.2)
    model.invoke("hello", temperature=0.9)
```

### 3. Custom Normalizers

For complex normalization or content redaction:

```python
def redact_dates(request_dict):
    # Redact dynamic inputs or strip timestamps
    return request_dict

with cassette("tests/cassettes", normalizer=redact_dates):
    model.invoke(...)
```

---

## Command Line Interface (CLI)

LLMCassette comes with a CLI tool to manage your cassettes.

### Stats
Show the number of cassettes, total size on disk, and estimated API latency saved:
```bash
llmcassette stats --path ./tests/cassettes
```

### Inspect
List all stored cassettes, their model, provider, and when they were recorded:
```bash
llmcassette inspect --path ./tests/cassettes
```

### Clean
Clean dynamic fields (`latency`, `created_at`) from cassettes to prevent noisy git diffs:
```bash
llmcassette clean --path ./tests/cassettes --remove-latency --remove-timestamps
```

---

## License

MIT License.
