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
- 🧹 CLI utilities

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