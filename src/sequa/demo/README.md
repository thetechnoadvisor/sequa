# Sequa Framework Demos

This directory contains standalone, production-ready framework integration demos for `sequa`.

## Available Framework Demos

- **LangGraph** (`langgraph_demo.py`):
  - StateGraph agent recording and deterministic offline replaying.
  - Multi-turn tool-calling agent with `bind_tools`.
  - PostgreSQL database storage backend (`PostgresStorage`).
- **OpenAI** (*Coming soon*)
- **CrewAI** (*Coming soon*)
- **AutoGen** (*Coming soon*)
- **Google ADK** (*Coming soon*)
- **Pydantic AI** (*Coming soon*)

## Running the Demos

To run the LangGraph demo:

```bash
python -m sequa.demo.langgraph_demo
```

Or programmatically in Python:

```python
from sequa.demo import run_langgraph_demo

run_langgraph_demo()
```
