"""Sequa LangGraph Integration Demo.

Demonstrates snapshot testing (recording and replaying) for LangGraph state graphs and agents
using Sequa's decorator / context manager with file and PostgreSQL storage backends.
"""

from __future__ import annotations

import os
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from sequa import cassette, PostgresStorage, MemoryStorage


class AgentState(TypedDict):
    """LangGraph agent state containing message history."""
    messages: Annotated[list[BaseMessage], operator.add]


@tool
def calculate_summary(topic: str, word_count: int = 50) -> str:
    """Tool to summarize a technical topic within specified word count limits."""
    return f"Summary of {topic}: Snapshot testing records LLM inputs/outputs for fast offline replay."


def demo_basic_langgraph_flow(cassette_dir: str = "demo_cassettes/langgraph_basic") -> None:
    """Demo 1: Basic LangGraph Agent State Graph (Record & Instant Offline Replay)."""
    print("\n=======================================================")
    print("  Demo 1: LangGraph State Graph Record & Replay")
    print("=======================================================")

    # 1. Build LangGraph State Graph
    api_key = os.getenv("GROQ_API_KEY") or "gsk_dummy"
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

    def chatbot_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    prompt = [HumanMessage(content="Explain LangGraph in 1 short sentence.")]

    # 2. Record Mode: Execute graph live and save interaction to cassette
    print("\n[Record Mode] Invoking LangGraph graph live...")
    with cassette(cassette_dir, mode="record"):
        result = graph.invoke({"messages": prompt})
        live_output = result["messages"][-1].content
        print("  -> Live Response:", live_output)

    # 3. Replay Mode: Replay from cassette with offline model (invalid API key)
    print("\n[Replay Mode] Replaying LangGraph response from cassette...")
    offline_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="invalid_key_offline")

    def offline_chatbot_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [offline_llm.invoke(state["messages"])]}

    builder_offline = StateGraph(AgentState)
    builder_offline.add_node("chatbot", offline_chatbot_node)
    builder_offline.add_edge(START, "chatbot")
    builder_offline.add_edge("chatbot", END)
    graph_offline = builder_offline.compile()

    with cassette(cassette_dir, mode="replay"):
        replayed = graph_offline.invoke({"messages": prompt})
        replayed_output = replayed["messages"][-1].content
        print("  -> Replayed Response:", replayed_output)
        assert replayed_output == live_output
        print("  ✓ SUCCESS: Replayed response matches recorded output exactly!")


def demo_tool_calling_langgraph_flow(cassette_dir: str = "demo_cassettes/langgraph_tools") -> None:
    """Demo 2: Tool-Calling LangGraph Agent (Record & Replay Tool Calls)."""
    print("\n=======================================================")
    print("  Demo 2: LangGraph Tool-Calling Agent Record & Replay")
    print("=======================================================")

    api_key = os.getenv("GROQ_API_KEY") or "gsk_dummy"
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key).bind_tools([calculate_summary])

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    prompt = [HumanMessage(content="Summarize snapshot testing using calculate_summary tool.")]

    print("\n[Record Mode] Invoking tool-bound LangGraph agent...")
    with cassette(cassette_dir, mode="record"):
        res = graph.invoke({"messages": prompt})
        tool_calls = getattr(res["messages"][-1], "tool_calls", [])
        print("  -> Recorded Tool Calls:", tool_calls)

    print("\n[Replay Mode] Replaying tool calls offline...")
    offline_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="invalid_key").bind_tools([calculate_summary])

    def offline_agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [offline_llm.invoke(state["messages"])]}

    builder_offline = StateGraph(AgentState)
    builder_offline.add_node("agent", offline_agent_node)
    builder_offline.add_edge(START, "agent")
    builder_offline.add_edge("agent", END)
    graph_offline = builder_offline.compile()

    with cassette(cassette_dir, mode="replay"):
        replayed_res = graph_offline.invoke({"messages": prompt})
        replayed_tool_calls = getattr(replayed_res["messages"][-1], "tool_calls", [])
        print("  -> Replayed Tool Calls:", replayed_tool_calls)
        assert len(replayed_tool_calls) > 0
        print("  ✓ SUCCESS: Tool call arguments replayed offline without API call!")


def demo_postgres_langgraph_flow() -> None:
    """Demo 3: LangGraph Agent with PostgreSQL Cassette Storage Backend."""
    print("\n=======================================================")
    print("  Demo 3: LangGraph Agent with PostgreSQL Storage")
    print("=======================================================")

    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "postgresql://postgres:password@localhost:5432/obs_db"
    os.environ["DATABASE_URL"] = db_url

    try:
        import psycopg
        conn = psycopg.connect(db_url, connect_timeout=2)
        conn.close()
    except Exception as e:
        print(f"  [Skipped] PostgreSQL database not reachable at {db_url}: {e}")
        return

    pg_storage = PostgresStorage(db_url=db_url)
    api_key = os.getenv("GROQ_API_KEY") or "gsk_dummy"
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

    def chatbot_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    prompt = [HumanMessage(content="What is PostgreSQL in 1 short sentence?")]
    cassette_key = "langgraph_postgres_demo"

    print(f"\n[Record Mode] Saving LangGraph cassette directly to PostgreSQL ({db_url})...")
    with cassette(cassette_key, mode="record", storage=pg_storage):
        result = graph.invoke({"messages": prompt})
        print("  -> Live Response:", result["messages"][-1].content)

    print("\n[PostgreSQL Query] Verifying cassette presence in PostgreSQL 'sequa_cassettes' table...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, provider, created_at FROM sequa_cassettes WHERE id LIKE %s", (f"%{cassette_key}%",))
            rows = cur.fetchall()
            for r in rows:
                print(f"  - DB Record ID: {r[0]} | Provider: {r[1]}")

    print("\n[Replay Mode] Replaying LangGraph from PostgreSQL backend...")
    offline_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="invalid_key")

    def offline_chatbot_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [offline_llm.invoke(state["messages"])]}

    builder_offline = StateGraph(AgentState)
    builder_offline.add_node("chatbot", offline_chatbot_node)
    builder_offline.add_edge(START, "chatbot")
    builder_offline.add_edge("chatbot", END)
    graph_offline = builder_offline.compile()

    with cassette(cassette_key, mode="replay", storage=pg_storage):
        replayed = graph_offline.invoke({"messages": prompt})
        print("  -> Replayed Output from Postgres:", replayed["messages"][-1].content)
        print("  ✓ SUCCESS: LangGraph state graph replayed directly from PostgreSQL!")


def run_langgraph_demo() -> None:
    """Run all LangGraph integration demos."""
    load_dotenv()
    print("=======================================================")
    print("          Sequa LangGraph Framework Demo               ")
    print("=======================================================")

    demo_basic_langgraph_flow()
    demo_tool_calling_langgraph_flow()
    demo_postgres_langgraph_flow()

    print("\n=======================================================")
    print("  All LangGraph Demos Completed Successfully!          ")
    print("=======================================================\n")


if __name__ == "__main__":
    run_langgraph_demo()
