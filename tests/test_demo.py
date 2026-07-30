import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from sequa.demo.langgraph_demo import demo_basic_langgraph_flow, demo_tool_calling_langgraph_flow


def test_langgraph_demo_flow(tmp_path):
    mock_msg = AIMessage(
        content="LangGraph is a graph-based framework.",
        tool_calls=[{"name": "calculate_summary", "args": {"topic": "snapshot testing", "word_count": 50}, "id": "tc_1", "type": "tool_call"}]
    )

    with patch.object(ChatGroq, "invoke", return_value=mock_msg):
        cassette_dir = str(tmp_path / "langgraph_test_cassette")
        demo_basic_langgraph_flow(cassette_dir=cassette_dir)
        demo_tool_calling_langgraph_flow(cassette_dir=str(tmp_path / "langgraph_tools_test"))
