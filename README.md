# Sequa 📼

Sequa is snapshot testing for LLM applications. Record once, replay forever.

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
from sequa import cassette

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
- **Streaming Support**: Full support for recording and replaying streaming responses (both sync and async generators).
- **PII & Sensitive Information Masking**: Automatically mask emails, phone numbers, credit cards, SSNs, IP addresses, API keys, and bearer tokens from cassettes.
- **Robust Key Sorting & Hashing**: Recursively sorts request inputs to generate deterministic hashes.
- **Custom Ignored Fields**: Easily ignore dynamic/unstable fields (e.g. `temperature`, `max_tokens`).
- **Custom Normalizers**: Redact, replace, or clean requests prior to hashing.
- **CLI Utilities**: Inspect, format, and calculate statistics of stored cassettes.

---

## Installation

Install Sequa from PyPI:

```bash
pip install sequa
```

Or using `uv`:

```bash
uv add sequa
```

For local development:

```bash
uv pip install -e .
```

---

## Configuration & Advanced API

### 1. Execution Modes

Control Sequa behavior via the `mode` parameter:

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

### 4. PII & Sensitive Information Masking

Automatically mask sensitive information like emails, phone numbers, IP addresses, and API keys inside request and response payloads before writing them to the cassette files.

To enable, set `mask_pii=True`:

```python
with cassette("tests/cassettes", mask_pii=True):
    # Any email, phone number, API key, etc. will be redacted in the cassette
    response = model.invoke("Send email to alice@example.com")
```

The matching cassette file will look like:
```json
{
  "request": {
    "messages": [
      {
        "role": "user",
        "content": "Send email to [EMAIL]"
      }
    ]
  },
  "response": { ... }
}
```

Masked patterns include:
- **Emails** (replaced by `[EMAIL]`)
- **Phone Numbers** (replaced by `[PHONE]`)
- **Credit Cards** (replaced by `[CREDIT_CARD]`)
- **Social Security Numbers** (replaced by `[SSN]`)
- **IP Addresses** (replaced by `[IP_ADDRESS]`)
- **API Keys / Secrets** (replaced by `[API_KEY]`)
- **Bearer Tokens** (replaced by `Bearer [TOKEN]`)

### 5. Streaming & Async Support

Sequa supports streaming responses (both sync and async generators) for OpenAI, Anthropic, and LangChain. The streaming chunks are captured on recording and replayed deterministically.

```python
# Streaming with OpenAI
from openai import OpenAI
from sequa.llm.adapters import OpenAIAdapter

client = OpenAI()
adapter = OpenAIAdapter()

with cassette("tests/cassettes", adapter=adapter):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a poem"}],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
```

---

## Command Line Interface (CLI)

Sequa comes with a CLI tool to manage your cassettes.

### Stats
Show the number of cassettes, total size on disk, and estimated API latency saved:
```bash
sequa stats --path ./tests/cassettes
```

### Inspect
List all stored cassettes, their model, provider, and when they were recorded:
```bash
sequa inspect --path ./tests/cassettes
```

### Clean
Clean dynamic fields (`latency`, `created_at`) from cassettes to prevent noisy git diffs:
```bash
sequa clean --path ./tests/cassettes --remove-latency --remove-timestamps
```

---

## License

MIT License.
