"""Unit and integration tests for Phase 2 AI Regression Testing Engine in Sequa."""

import json
import pytest
from sequa import (
    Cassette,
    CostDiff,
    LatencyDiff,
    PromptDiff,
    RegressionError,
    RegressionReport,
    SemanticDiff,
    ToolDiff,
    compare_executions,
    cassette,
)
from sequa.cli.main import main


@pytest.fixture
def mock_cassette_pair():
    old_data = {
        "id": "old-cas-101",
        "provider": "openai",
        "hash": "hash_old_101",
        "request": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "You are a helpful customer support AI."},
                {"role": "user", "content": "How do I request a refund for my order?"},
            ],
        },
        "response": {
            "output": "To request a refund, please log in to your account, navigate to Orders, and click Request Refund.",
            "choices": [
                {
                    "message": {
                        "content": "To request a refund, please log in to your account, navigate to Orders, and click Request Refund.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup_order",
                                    "arguments": json.dumps({"order_id": "ORD-123"}),
                                }
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        },
        "metadata": {"duration_ms": 350.0, "latency_ms": 350.0},
    }

    new_data = {
        "id": "new-cas-202",
        "provider": "openai",
        "hash": "hash_new_202",
        "request": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are an expert customer support assistant."},
                {"role": "user", "content": "How do I request a refund for my order?"},
            ],
        },
        "response": {
            "output": "To initiate a refund, log into your account dashboard, view your purchase history, select your order, and submit a refund claim.",
            "choices": [
                {
                    "message": {
                        "content": "To initiate a refund, log into your account dashboard, view your purchase history, select your order, and submit a refund claim.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup_order",
                                    "arguments": json.dumps({"order_id": "ORD-123", "include_items": True}),
                                }
                            },
                            {
                                "function": {
                                    "name": "check_eligibility",
                                    "arguments": json.dumps({"order_id": "ORD-123"}),
                                }
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 65, "completion_tokens": 25, "total_tokens": 90},
        },
        "metadata": {"duration_ms": 280.0, "latency_ms": 280.0},
    }

    return old_data, new_data


def test_compare_executions_all_dimensions(mock_cassette_pair):
    old_data, new_data = mock_cassette_pair
    report = compare_executions(old_data, new_data)

    assert isinstance(report, RegressionReport)
    assert report.old_id == "old-cas-101"
    assert report.new_id == "new-cas-202"
    assert report.has_changes is True

    # 1. Prompt Diff
    p_diff = report.prompt_diff
    assert isinstance(p_diff, PromptDiff)
    assert p_diff.has_changes is True
    assert "model" in p_diff.param_changes
    assert p_diff.param_changes["model"] == {"old": "gpt-4o-mini", "new": "gpt-4o"}
    assert p_diff.param_changes["temperature"] == {"old": 0.7, "new": 0.2}

    # 2. Tool Diff
    t_diff = report.tool_diff
    assert isinstance(t_diff, ToolDiff)
    assert t_diff.has_changes is True
    assert len(t_diff.added_tools) == 1
    assert t_diff.added_tools[0]["name"] == "check_eligibility"
    assert len(t_diff.modified_tools) == 1
    assert t_diff.modified_tools[0]["name"] == "lookup_order"

    # 3. Semantic Diff
    s_diff = report.semantic_diff
    assert isinstance(s_diff, SemanticDiff)
    assert 0.35 <= s_diff.similarity_score <= 0.95
    assert s_diff.status in ("MINOR_DIFF", "SIGNIFICANT_DRIFT")

    # 4. Cost Diff
    c_diff = report.cost_diff
    assert isinstance(c_diff, CostDiff)
    assert c_diff.old_tokens["total"] == 70
    assert c_diff.new_tokens["total"] == 90
    assert c_diff.token_delta["total"] == 20
    assert c_diff.new_cost_usd > c_diff.old_cost_usd

    # 5. Latency Diff
    l_diff = report.latency_diff
    assert isinstance(l_diff, LatencyDiff)
    assert l_diff.old_latency_ms == 350.0
    assert l_diff.new_latency_ms == 280.0
    assert l_diff.delta_ms == -70.0
    assert l_diff.percent_change < 0.0


def test_identical_executions_no_changes():
    data = {
        "id": "ident-1",
        "request": {"messages": [{"role": "user", "content": "Hello world"}]},
        "response": {
            "output": "Hello there!",
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        "metadata": {"duration_ms": 100.0},
    }

    report = compare_executions(data, data)
    assert report.semantic_diff.similarity_score == 1.0
    assert report.semantic_diff.status == "MATCH"
    assert report.prompt_diff.has_changes is False
    assert report.tool_diff.has_changes is False
    assert report.cost_diff.token_delta["total"] == 0
    assert report.latency_diff.delta_ms == 0.0


def test_regression_report_assertions(mock_cassette_pair):
    old_data, new_data = mock_cassette_pair
    report = compare_executions(old_data, new_data)

    # Threshold failure assertion
    with pytest.raises(RegressionError) as exc_info:
        report.assert_no_regression(similarity_threshold=0.99)
    assert "Semantic similarity score" in str(exc_info.value)
    assert exc_info.value.report is report

    # Tool changes failure assertion
    with pytest.raises(RegressionError) as exc_info:
        report.assert_no_regression(similarity_threshold=0.35, allow_tool_changes=False)
    assert "Tool call changes detected" in str(exc_info.value)

    # High similarity threshold pass
    report.assert_no_regression(similarity_threshold=0.35, allow_tool_changes=True)


def test_regression_report_renderers(mock_cassette_pair):
    old_data, new_data = mock_cassette_pair
    report = compare_executions(old_data, new_data)

    # 1. Text rendering
    text_out = report.render_text(use_color=False)
    assert "SEQUA PHASE 2 REGRESSION TEST REPORT" in text_out
    assert "1. 📝 PROMPT DIFF" in text_out
    assert "2. 🛠️ TOOL DIFF" in text_out
    assert "3. 🧠 SEMANTIC DIFF" in text_out
    assert "4. 💰 COST DIFF" in text_out
    assert "5. ⚡ LATENCY DIFF" in text_out

    # 2. Markdown rendering
    md_out = report.render_markdown()
    assert "# 📼 Sequa Phase 2 Regression Test Report" in md_out
    assert "## 1. 📝 Prompt Diff" in md_out
    assert "## 2. 🛠️ Tool Diff" in md_out
    assert "## 3. 🧠 Semantic Output Diff" in md_out
    assert "## 4. 💰 Cost & Token Diff" in md_out
    assert "## 5. ⚡ Latency Diff" in md_out

    # 3. HTML rendering
    html_out = report.render_html()
    assert "<html>" in html_out
    assert "Sequa Phase 2 Regression Test Report" in html_out

    # 4. Dict export
    d = report.to_dict()
    assert d["old_id"] == "old-cas-101"
    assert d["new_id"] == "new-cas-202"
    assert "prompt_diff" in d
    assert "tool_diff" in d
    assert "semantic_diff" in d
    assert "cost_diff" in d
    assert "latency_diff" in d


def test_mode_regression_with_cassette(tmp_path):
    cas_path = str(tmp_path / "test_reg.json")
    old_cas = Cassette(
        id="ref-cassette",
        provider="langchain_groq",
        request={"messages": [{"role": "user", "content": "What is the capital of France?"}]},
        response={"output": "The capital of France is Paris."},
        metadata={"duration_ms": 200.0},
    )
    old_cas.save(cas_path)

    # Running with mode="regression" loads reference cas_path and intercepts call
    with cassette(path=cas_path, mode="regression") as cas:
        from langchain_core.messages import AIMessage

        def mock_call():
            return AIMessage(content="Paris is the capital city of France.")

        res = cas.intercept(mock_call)
        assert res["raw"].content == "Paris is the capital city of France."

        report = cas.regression_report
        assert report is not None
        assert report.old_id in (cas_path, "ref-cassette")
        assert report.semantic_diff.similarity_score > 0.6
        assert report.latency_diff.old_latency_ms == 200.0


def test_cli_regression_command(tmp_path, capsys):
    f1 = tmp_path / "cas1.json"
    f2 = tmp_path / "cas2.json"

    d1 = {
        "id": "c1",
        "request": {"messages": [{"role": "user", "content": "Explain AI"}]},
        "response": {"output": "AI stands for Artificial Intelligence."},
        "metadata": {"duration_ms": 150.0},
    }
    d2 = {
        "id": "c2",
        "request": {"messages": [{"role": "user", "content": "Explain AI"}]},
        "response": {"output": "Artificial Intelligence is machine intelligence."},
        "metadata": {"duration_ms": 120.0},
    }

    with open(f1, "w") as f:
        json.dump(d1, f)
    with open(f2, "w") as f:
        json.dump(d2, f)

    from sequa.cli.main import cmd_regression
    import argparse

    args = argparse.Namespace(
        execution_1=str(f1),
        execution_2=str(f2),
        path=str(tmp_path),
        format="text",
        output=None,
        fail_on_drift=False,
        threshold=0.85,
    )

    exit_code = cmd_regression(args)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SEQUA PHASE 2 REGRESSION TEST REPORT" in captured.out
    assert "Artificial Intelligence" in captured.out


def test_mode_regression_with_memory_storage():
    from sequa import MemoryStorage

    mem_storage = MemoryStorage()
    ref_cas = Cassette(
        id="mem-ref-1",
        provider="openai",
        request={"messages": [{"role": "user", "content": "Explain gravity"}]},
        response={"output": "Gravity is the force that attracts a body toward the center of the earth."},
        metadata={"duration_ms": 180.0},
    )
    mem_storage.save(ref_cas, cassette_id="mem_ref_key")

    with cassette(path="mem_ref_key", storage=mem_storage, mode="regression") as cas:
        from langchain_core.messages import AIMessage

        def mock_call():
            return AIMessage(content="Gravity pulls masses together.")

        res = cas.intercept(mock_call)
        assert res["raw"].content == "Gravity pulls masses together."

        report = cas.regression_report
        assert report is not None
        assert report.old_id == "mem-ref-1"
        assert report.semantic_diff.similarity_score > 0.1
        assert report.latency_diff.old_latency_ms == 180.0

