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
- **Tool Calling & Function Calling**: Full support for recording, hashing, and replaying tool calls across OpenAI, Anthropic, and LangChain models. Tool definitions, tool call requests, and tool call responses are stored deterministically in cassettes and reconstructed upon playback.
- **Streaming Support**: Full support for recording and replaying streaming responses (both sync and async generators).
- **PII & Sensitive Information Masking**: Automatically mask emails, phone numbers, credit cards, SSNs, IP addresses, API keys, and bearer tokens from cassettes.
- **NVIDIA NeMo Guardrails Integration**: Apply official NVIDIA NeMo Guardrails (`nemoguardrails`) on input prompts before LLM execution and on output responses after generation. Selectable input and output guardrails.
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

### 6. NVIDIA NeMo Guardrails Integration

Sequa integrates the official **NVIDIA NeMo Guardrails** (`nemoguardrails`) package to evaluate input prompts before sending them to the LLM and output responses after generation. Users can select which guardrails to enable via the `guardrails` parameter in `cassette()`.

```python
from langchain_groq import ChatGroq
from sequa import cassette

model = ChatGroq(model_name="llama-3.1-8b-instant")

# Enable input jailbreak detection and output hallucination checking
with cassette("tests/cassettes", guardrails=["input_jailbreak", "output_hallucination"]):
    # 1. Input prompt is evaluated before calling LLM:
    # If flagged as jailbreak, LLM call is blocked immediately.
    response = model.invoke("Explain how photosynthesis works.")

    # 2. Output response is evaluated after generation:
    # If output contains hallucination, output response is blocked.
```

Available Guardrails:
- **Input Guardrails**:
  - `"input_jailbreak"`: Detects prompt injection, system prompt override, or DAN mode attempts.
  - `"input_moderation"`: Detects harmful, unsafe, or dangerous input prompts.
  - `"input_profanity"`: Filters profanity/obscenity in prompt input.
- **Output Guardrails**:
  - `"output_moderation"`: Detects harmful or toxic generated response text.
  - `"output_hallucination"`: Detects ungrounded or fabricated statements ("I am making this up").
  - `"output_profanity"`: Filters profanity in generated LLM responses.

### 7. Tool Calling & Function Calling

Sequa captures tool call definitions (`tools`, `tool_choice`), tool call outputs (`tool_calls`, `function_call`), and tool call response histories across OpenAI, Anthropic, and LangChain models.

```python
from openai import OpenAI
from sequa import cassette
from sequa.llm.adapters import OpenAIAdapter

client = OpenAI()
adapter = OpenAIAdapter()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get location weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

# 1. Record tool call interaction
with cassette("tests/cassettes/tools_flow", mode="record", adapter=adapter):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools
    )
    # Output tool call: response.choices[0].message.tool_calls[0].function.name -> "get_weather"

# 2. Replay instantly from cassette
with cassette("tests/cassettes/tools_flow", mode="replay", adapter=adapter):
    replayed = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools
    )
    print(replayed.choices[0].message.tool_calls[0].function.arguments)
```

### 8. Flexible Storage Spaces (File, Memory & PostgreSQL Backends)

Sequa supports modular storage backends via the `storage` parameter. You can specify string options (`"file"`, `"memory"`, `"postgres"`) or pass a `StorageBackend` instance (`FileStorage`, `MemoryStorage`, `PostgresStorage`).

- **`storage="file"`** (Default): Stores cassettes as JSON files on disk.
- **`storage="memory"`**: Stores cassettes in-memory without creating any files on disk — perfect for unit tests, CI pipelines, benchmarking, and quick prototyping.
- **`storage="postgres"`**: Stores cassettes centrally in a PostgreSQL database table (`sequa_cassettes`). Reads `DATABASE_URL` or `POSTGRES_URL` environment variables by default.

```python
from sequa import cassette, FileStorage, MemoryStorage, PostgresStorage, Cassette

# 1. File Storage: Store cassettes on disk as JSON files (Default)
with cassette("tests/cassettes", storage="file"):
    response = model.invoke("Hello file storage!")

# 2. Memory Storage: Store cassettes purely in RAM (zero disk files)
with cassette(storage="memory"):
    response = model.invoke("Hello in-memory storage!")

# 3. PostgreSQL Storage: Store cassettes in a Postgres database
pg_storage = PostgresStorage(db_url="postgresql://user:pass@localhost:5432/mydb")
with cassette(storage=pg_storage):
    response = model.invoke("Hello Postgres storage!")

# 4. Programmatic Cassette object with custom storage backend
cas = Cassette(
    request={"messages": [{"role": "user", "content": "hi"}]},
    response={"output": "hello"},
    storage=pg_storage
)
cas.save(path_or_id="pg_custom_key")
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
