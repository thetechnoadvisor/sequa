# Contributing to Sequa 📼

Thank you for your interest in contributing to **Sequa**! We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, new LLM provider adapters, and code contributions.

This document provides a set of guidelines and instructions to help you get started with contributing to Sequa.

---

## 📜 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Adding LLM Provider Adapters](#adding-llm-provider-adapters)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
- [Development & Testing Workflow](#-development--testing-workflow)
  - [Project Structure](#project-structure)
  - [Running Tests](#running-tests)
  - [Code Style & Guidelines](#code-style--guidelines)
- [Submitting a Pull Request](#-submitting-a-pull-request)
- [License](#-license)

---

## 🤝 Code of Conduct

We are committed to providing a welcoming, inclusive, and respectful community for everyone. Please maintain a professional and courteous tone in all discussions, issues, and pull requests.

---

##💡 How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing GitHub issues to make sure the problem hasn't already been reported.

When creating a bug report, please include:
- A clear and descriptive title.
- Steps to reproduce the issue.
- Python version (`python --version`) and Sequa version.
- LLM provider or SDK version (e.g. `openai`, `anthropic`, `langchain-groq`).
- Expected vs. actual behavior.
- Code snippet or minimal reproducible example.

### Suggesting Features

We welcome ideas for new features or improvements (e.g., new execution modes, enhanced PII masking patterns, new CLI tools).

When submitting a feature request:
- Explain **why** this feature would be useful to users.
- Describe **how** you envision it working.
- Provide example code or API usage if applicable.

### Adding LLM Provider Adapters

Sequa is designed to support multiple LLM providers. If your favorite provider or framework (e.g. Cohere, Google Gemini, Ollama, Mistral) isn't yet supported, we welcome contributions!
- Place new adapters under `src/sequa/llm/adapters/`.
- Ensure streaming (both sync and async generators) and non-streaming responses are handled properly.
- Include corresponding unit and integration tests in `tests/`.

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.12+**
- **uv** (Recommended Fast Python package installer and resolver):
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Local Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/sequa.git
   cd sequa
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Sequa in editable mode:**
   ```bash
   uv pip install -e .
   ```

---

## 🧪 Development & Testing Workflow

### Project Structure

```
llmcassette/
├── src/
│   └── sequa/
│       ├── __init__.py
│       ├── cassette.py         # Main cassette context manager & top-level API
│       ├── recorder.py         # Recording & interceptor engine
│       ├── matcher.py          # Request matching, hashing, ignore fields & PII masking
│       ├── storage.py          # Cassette disk serialization & loading
│       ├── models.py           # Data structures for requests, responses & cassettes
│       ├── utils.py            # Hashing and helper utilities
│       ├── cli/                # Command-line interface logic (stats, inspect, clean)
│       └── llm/                # Provider adapters (OpenAI, Anthropic, LangChain, etc.)
├── tests/                      # Pytest test suite
│   ├── test_adapters.py
│   ├── test_cli.py
│   ├── test_integration.py
│   ├── test_masking.py
│   ├── test_matcher.py
│   ├── test_openai_anthropic.py
│   ├── test_storage.py
│   └── test_streaming.py
├── pyproject.toml              # Build configuration & dependencies
└── README.md                   # Project documentation
```

### Running Tests

We use [pytest](https://docs.pytest.org/) for testing. Make sure all tests pass before opening a pull request.

Run the test suite using `uv`:

```bash
uv run pytest
```

To run a specific test file:
```bash
uv run pytest tests/test_masking.py
```

### Code Style & Guidelines

- **Type Hints:** Use explicit Python type annotations (`mypy` style) for public functions and classes.
- **Clean Git Diffs:** Run `sequa clean` on any cassettes generated during tests to prevent unnecessary timestamp or latency diffs.
- **Docstrings:** Use clear Google or standard Python docstrings for public functions and modules.

---

## 🚀 Submitting a Pull Request

1. **Create a new feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Commit your changes:**
   Keep commit messages descriptive and concise:
   ```bash
   git commit -m "feat(adapter): add support for Gemini streaming responses"
   ```

3. **Run tests:** Ensure the full test suite passes (`uv run pytest`).

4. **Push to your fork:**
   ```bash
   git push origin feat/your-feature-name
   ```

5. **Open a Pull Request:**
   - Describe what changed and why.
   - Reference any relevant issues (e.g. `Closes #42`).

---

## 📄 License

By contributing to Sequa, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
