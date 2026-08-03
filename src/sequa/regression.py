"""Phase 2 AI Regression Testing Engine for Sequa.

Compares an Old Cassette (reference run) against a New Execution (live or updated run)
across 5 key analysis dimensions:
1. Prompt Diff (system/user messages & parameters)
2. Tool Diff (added, removed, modified tool calls & arguments)
3. Semantic Diff (text output similarity score, status, unified diff)
4. Cost Diff (token usage deltas & estimated USD cost change)
5. Latency Diff (duration in ms & percentage change)
"""

from __future__ import annotations

import copy
import datetime
import difflib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sequa.models import Cassette


class SequaError(Exception):
    """Base exception for Sequa."""


class RegressionError(SequaError):
    """Exception raised when a regression test fails threshold or assertion checks."""

    def __init__(self, message: str, report: RegressionReport | None = None) -> None:
        super().__init__(message)
        self.report = report


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_name_key: (prompt_cost_per_1k_tokens, completion_cost_per_1k_tokens)
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-opus": (0.015, 0.075),
    "llama-3": (0.0002, 0.0002),
    "llama-3.1": (0.0002, 0.0002),
    "llama-3.2": (0.00015, 0.00015),
    "mixtral": (0.0006, 0.0006),
}
DEFAULT_PRICING = (0.0015, 0.0020)


@dataclass
class PromptDiff:
    has_changes: bool
    system_prompt_diff: str
    user_prompt_diff: str
    param_changes: dict[str, dict[str, Any]]
    old_system_prompt: str = ""
    new_system_prompt: str = ""
    old_user_prompt: str = ""
    new_user_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_changes": self.has_changes,
            "system_prompt_diff": self.system_prompt_diff,
            "user_prompt_diff": self.user_prompt_diff,
            "param_changes": self.param_changes,
            "old_system_prompt": self.old_system_prompt,
            "new_system_prompt": self.new_system_prompt,
            "old_user_prompt": self.old_user_prompt,
            "new_user_prompt": self.new_user_prompt,
        }


@dataclass
class ToolDiff:
    has_changes: bool
    added_tools: list[dict[str, Any]]
    removed_tools: list[dict[str, Any]]
    modified_tools: list[dict[str, Any]]
    order_changed: bool
    old_tools: list[dict[str, Any]]
    new_tools: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_changes": self.has_changes,
            "added_tools": self.added_tools,
            "removed_tools": self.removed_tools,
            "modified_tools": self.modified_tools,
            "order_changed": self.order_changed,
            "old_tools": self.old_tools,
            "new_tools": self.new_tools,
        }


@dataclass
class SemanticDiff:
    similarity_score: float
    status: str  # "MATCH", "MINOR_DIFF", "SIGNIFICANT_DRIFT"
    diff_text: str
    old_text: str
    new_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity_score": self.similarity_score,
            "status": self.status,
            "diff_text": self.diff_text,
            "old_text": self.old_text,
            "new_text": self.new_text,
        }


@dataclass
class CostDiff:
    old_tokens: dict[str, int]
    new_tokens: dict[str, int]
    token_delta: dict[str, int]
    old_cost_usd: float
    new_cost_usd: float
    cost_delta_usd: float
    cost_change_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_tokens": self.old_tokens,
            "new_tokens": self.new_tokens,
            "token_delta": self.token_delta,
            "old_cost_usd": round(self.old_cost_usd, 6),
            "new_cost_usd": round(self.new_cost_usd, 6),
            "cost_delta_usd": round(self.cost_delta_usd, 6),
            "cost_change_percent": round(self.cost_change_percent, 2),
        }


@dataclass
class LatencyDiff:
    old_latency_ms: float
    new_latency_ms: float
    delta_ms: float
    percent_change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_latency_ms": round(self.old_latency_ms, 2),
            "new_latency_ms": round(self.new_latency_ms, 2),
            "delta_ms": round(self.delta_ms, 2),
            "percent_change": round(self.percent_change, 2),
        }


@dataclass
class RegressionReport:
    old_id: str
    new_id: str
    prompt_diff: PromptDiff
    tool_diff: ToolDiff
    semantic_diff: SemanticDiff
    cost_diff: CostDiff
    latency_diff: LatencyDiff
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    @property
    def has_changes(self) -> bool:
        return (
            self.prompt_diff.has_changes
            or self.tool_diff.has_changes
            or self.semantic_diff.similarity_score < 0.99
            or self.cost_diff.token_delta["total"] != 0
            or abs(self.latency_diff.delta_ms) > 10.0
        )

    def assert_no_regression(
        self,
        similarity_threshold: float = 0.85,
        allow_tool_changes: bool = False,
        max_cost_increase_pct: float | None = None,
        max_latency_increase_pct: float | None = None,
    ) -> None:
        """Enforces test assertions against regression metrics."""
        errors: list[str] = []

        if self.semantic_diff.similarity_score < similarity_threshold:
            errors.append(
                f"Semantic similarity score {self.semantic_diff.similarity_score:.4f} is below "
                f"threshold {similarity_threshold:.4f} (Status: {self.semantic_diff.status})."
            )

        if not allow_tool_changes and self.tool_diff.has_changes:
            desc = []
            if self.tool_diff.added_tools:
                desc.append(f"added {len(self.tool_diff.added_tools)} tool(s)")
            if self.tool_diff.removed_tools:
                desc.append(f"removed {len(self.tool_diff.removed_tools)} tool(s)")
            if self.tool_diff.modified_tools:
                desc.append(f"modified {len(self.tool_diff.modified_tools)} tool call(s)")
            if self.tool_diff.order_changed:
                desc.append("tool execution order changed")
            errors.append(f"Tool call changes detected: {', '.join(desc)}.")

        if max_cost_increase_pct is not None:
            if self.cost_diff.cost_change_percent > max_cost_increase_pct:
                errors.append(
                    f"Cost increased by {self.cost_diff.cost_change_percent:.1f}%, exceeding "
                    f"limit of {max_cost_increase_pct:.1f}%."
                )

        if max_latency_increase_pct is not None:
            if self.latency_diff.percent_change > max_latency_increase_pct:
                errors.append(
                    f"Latency increased by {self.latency_diff.percent_change:.1f}%, exceeding "
                    f"limit of {max_latency_increase_pct:.1f}%."
                )

        if errors:
            msg = "Regression test failed:\n  - " + "\n  - ".join(errors)
            raise RegressionError(msg, report=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_id": self.old_id,
            "new_id": self.new_id,
            "created_at": self.created_at,
            "has_changes": self.has_changes,
            "prompt_diff": self.prompt_diff.to_dict(),
            "tool_diff": self.tool_diff.to_dict(),
            "semantic_diff": self.semantic_diff.to_dict(),
            "cost_diff": self.cost_diff.to_dict(),
            "latency_diff": self.latency_diff.to_dict(),
        }

    def render_text(self, use_color: bool = True) -> str:
        """Render beautiful terminal report."""
        c_reset = "\033[0m" if use_color else ""
        c_bold = "\033[1m" if use_color else ""
        c_cyan = "\033[36m" if use_color else ""
        c_green = "\033[32m" if use_color else ""
        c_yellow = "\033[33m" if use_color else ""
        c_red = "\033[31m" if use_color else ""
        c_blue = "\033[34m" if use_color else ""

        status_str = (
            f"{c_green}PASSED (Match){c_reset}"
            if self.semantic_diff.status == "MATCH"
            else f"{c_yellow}WARNING (Minor Diff){c_reset}"
            if self.semantic_diff.status == "MINOR_DIFF"
            else f"{c_red}FAILED (Drift Detected){c_reset}"
        )

        lines = [
            f"{c_bold}══════════════════════════════════════════════════════════════════════════════{c_reset}",
            f"{c_bold}                 SEQUA PHASE 2 REGRESSION TEST REPORT                         {c_reset}",
            f"{c_bold}══════════════════════════════════════════════════════════════════════════════{c_reset}",
            f"  Reference Cassette : {self.old_id}",
            f"  New Execution      : {self.new_id}",
            f"  Evaluation Status  : {status_str}",
            f"──────────────────────────────────────────────────────────────────────────────",
            f"{c_bold}1. 📝 PROMPT DIFF{c_reset}",
        ]

        if not self.prompt_diff.has_changes:
            lines.append("   ✓ Prompts and execution parameters identical")
        else:
            if self.prompt_diff.param_changes:
                lines.append("   Parameter Changes:")
                for k, v in self.prompt_diff.param_changes.items():
                    lines.append(f"     - {k}: {v['old']} ➔ {v['new']}")
            if self.prompt_diff.system_prompt_diff:
                lines.append("   System Prompt Diff:")
                for line in self.prompt_diff.system_prompt_diff.splitlines()[:5]:
                    lines.append(f"     {line}")
            if self.prompt_diff.user_prompt_diff:
                lines.append("   User Prompt Diff:")
                for line in self.prompt_diff.user_prompt_diff.splitlines()[:5]:
                    lines.append(f"     {line}")

        lines.extend([
            f"──────────────────────────────────────────────────────────────────────────────",
            f"{c_bold}2. 🛠️ TOOL DIFF{c_reset}",
        ])

        if not self.tool_diff.has_changes:
            lines.append("   ✓ Tool calls identical")
        else:
            if self.tool_diff.added_tools:
                lines.append(f"   + Added Tools ({len(self.tool_diff.added_tools)}):")
                for t in self.tool_diff.added_tools:
                    lines.append(f"     + {t.get('name')}({t.get('args', {})})")
            if self.tool_diff.removed_tools:
                lines.append(f"   - Removed Tools ({len(self.tool_diff.removed_tools)}):")
                for t in self.tool_diff.removed_tools:
                    lines.append(f"     - {t.get('name')}({t.get('args', {})})")
            if self.tool_diff.modified_tools:
                lines.append(f"   ~ Modified Tools ({len(self.tool_diff.modified_tools)}):")
                for m in self.tool_diff.modified_tools:
                    lines.append(f"     ~ {m.get('name')}: args changed")

        score_color = (
            c_green if self.semantic_diff.similarity_score >= 0.95
            else c_yellow if self.semantic_diff.similarity_score >= 0.85
            else c_red
        )
        lines.extend([
            f"──────────────────────────────────────────────────────────────────────────────",
            f"{c_bold}3. 🧠 SEMANTIC DIFF{c_reset}",
            f"   Similarity Score : {score_color}{self.semantic_diff.similarity_score * 100:.1f}%{c_reset} ({self.semantic_diff.similarity_score:.4f})",
            f"   Status           : {self.semantic_diff.status}",
        ])

        if self.semantic_diff.diff_text and self.semantic_diff.similarity_score < 1.0:
            lines.append("   Output Diff:")
            for line in self.semantic_diff.diff_text.splitlines()[:10]:
                if use_color:
                    if line.startswith("+"):
                        lines.append(f"     {c_green}{line}{c_reset}")
                    elif line.startswith("-"):
                        lines.append(f"     {c_red}{line}{c_reset}")
                    else:
                        lines.append(f"     {line}")
                else:
                    lines.append(f"     {line}")

        cost_sign = "+" if self.cost_diff.cost_delta_usd > 0 else ""
        token_sign = "+" if self.cost_diff.token_delta["total"] > 0 else ""
        lines.extend([
            f"──────────────────────────────────────────────────────────────────────────────",
            f"{c_bold}4. 💰 COST DIFF{c_reset}",
            f"   Old Tokens       : {self.cost_diff.old_tokens['total']} (P: {self.cost_diff.old_tokens['prompt']} / C: {self.cost_diff.old_tokens['completion']})",
            f"   New Tokens       : {self.cost_diff.new_tokens['total']} (P: {self.cost_diff.new_tokens['prompt']} / C: {self.cost_diff.new_tokens['completion']})",
            f"   Token Delta      : {token_sign}{self.cost_diff.token_delta['total']} total ({self.cost_diff.cost_change_percent:+.1f}%)",
            f"   Est. Cost Delta  : {cost_sign}${self.cost_diff.cost_delta_usd:.6f} USD",
        ])

        lat_sign = "+" if self.latency_diff.delta_ms > 0 else ""
        lat_color = c_green if self.latency_diff.delta_ms <= 0 else c_yellow
        lines.extend([
            f"──────────────────────────────────────────────────────────────────────────────",
            f"{c_bold}5. ⚡ LATENCY DIFF{c_reset}",
            f"   Old Duration     : {self.latency_diff.old_latency_ms:.1f} ms",
            f"   New Duration     : {self.latency_diff.new_latency_ms:.1f} ms",
            f"   Latency Delta    : {lat_color}{lat_sign}{self.latency_diff.delta_ms:.1f} ms ({self.latency_diff.percent_change:+.1f}%){c_reset}",
            f"══════════════════════════════════════════════════════════════════════════════",
        ])

        return "\n".join(lines) + "\n"

    def render_markdown(self) -> str:
        """Render clean GitHub Markdown regression report."""
        status_badge = (
            "🟢 **PASSED**"
            if self.semantic_diff.status == "MATCH"
            else "🟡 **WARNING (Minor Diff)**"
            if self.semantic_diff.status == "MINOR_DIFF"
            else "🔴 **FAILED (Drift Detected)**"
        )

        md = [
            "# 📼 Sequa Phase 2 Regression Test Report",
            "",
            f"**Status:** {status_badge}",
            "",
            "| Metric | Reference (Old) | New Execution | Change / Delta |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Cassette ID / Path** | `{self.old_id}` | `{self.new_id}` | - |",
            f"| **Semantic Similarity** | 100% | `{self.semantic_diff.similarity_score * 100:.1f}%` | **{self.semantic_diff.status}** |",
            f"| **Total Tokens** | {self.cost_diff.old_tokens['total']} | {self.cost_diff.new_tokens['total']} | `{self.cost_diff.token_delta['total']:+} tokens` |",
            f"| **Est. Cost (USD)** | `${self.cost_diff.old_cost_usd:.6f}` | `${self.cost_diff.new_cost_usd:.6f}` | `${self.cost_diff.cost_delta_usd:+.6f}` (`{self.cost_diff.cost_change_percent:+.1f}%`) |",
            f"| **Latency** | `{self.latency_diff.old_latency_ms:.1f} ms` | `{self.latency_diff.new_latency_ms:.1f} ms` | `{self.latency_diff.delta_ms:+.1f} ms` (`{self.latency_diff.percent_change:+.1f}%`) |",
            "",
            "## 1. 📝 Prompt Diff",
            "",
        ]

        if not self.prompt_diff.has_changes:
            md.append("_No prompt or parameter changes detected._\n")
        else:
            if self.prompt_diff.param_changes:
                md.append("### Parameter Changes\n")
                md.append("| Parameter | Old Value | New Value |")
                md.append("| :--- | :--- | :--- |")
                for k, v in self.prompt_diff.param_changes.items():
                    md.append(f"| `{k}` | `{v['old']}` | `{v['new']}` |")
                md.append("")

            if self.prompt_diff.system_prompt_diff:
                md.append("### System Prompt Diff\n```diff")
                md.append(self.prompt_diff.system_prompt_diff)
                md.append("```\n")

            if self.prompt_diff.user_prompt_diff:
                md.append("### User Prompt Diff\n```diff")
                md.append(self.prompt_diff.user_prompt_diff)
                md.append("```\n")

        md.append("## 2. 🛠️ Tool Diff\n")
        if not self.tool_diff.has_changes:
            md.append("_No tool call changes detected._\n")
        else:
            if self.tool_diff.added_tools:
                md.append(f"**Added Tools ({len(self.tool_diff.added_tools)}):**")
                for t in self.tool_diff.added_tools:
                    md.append(f"- `{t.get('name')}`: `{json.dumps(t.get('args', {}))}`")
                md.append("")
            if self.tool_diff.removed_tools:
                md.append(f"**Removed Tools ({len(self.tool_diff.removed_tools)}):**")
                for t in self.tool_diff.removed_tools:
                    md.append(f"- `{t.get('name')}`: `{json.dumps(t.get('args', {}))}`")
                md.append("")
            if self.tool_diff.modified_tools:
                md.append(f"**Modified Tools ({len(self.tool_diff.modified_tools)}):**")
                for m in self.tool_diff.modified_tools:
                    md.append(f"- `{m.get('name')}` arguments modified")
                md.append("")

        md.append("## 3. 🧠 Semantic Output Diff\n")
        md.append(
            f"**Similarity Score:** `{self.semantic_diff.similarity_score * 100:.1f}%` ({self.semantic_diff.similarity_score:.4f})\n"
        )
        if self.semantic_diff.diff_text and self.semantic_diff.similarity_score < 1.0:
            md.append("```diff")
            md.append(self.semantic_diff.diff_text)
            md.append("```\n")
        else:
            md.append("_Output texts are semantically identical._\n")

        md.append("## 4. 💰 Cost & Token Diff\n")
        md.append(
            f"- **Prompt Tokens:** `{self.cost_diff.old_tokens['prompt']}` ➔ `{self.cost_diff.new_tokens['prompt']}` ({self.cost_diff.token_delta['prompt']:+d})\n"
            f"- **Completion Tokens:** `{self.cost_diff.old_tokens['completion']}` ➔ `{self.cost_diff.new_tokens['completion']}` ({self.cost_diff.token_delta['completion']:+d})\n"
            f"- **Total Tokens:** `{self.cost_diff.old_tokens['total']}` ➔ `{self.cost_diff.new_tokens['total']}` ({self.cost_diff.token_delta['total']:+d})\n"
            f"- **Estimated USD Cost:** `${self.cost_diff.old_cost_usd:.6f}` ➔ `${self.cost_diff.new_cost_usd:.6f}` (`${self.cost_diff.cost_delta_usd:+.6f}`)\n"
        )

        md.append("## 5. ⚡ Latency Diff\n")
        md.append(
            f"- **Old Latency:** `{self.latency_diff.old_latency_ms:.1f} ms`\n"
            f"- **New Latency:** `{self.latency_diff.new_latency_ms:.1f} ms`\n"
            f"- **Duration Change:** `{self.latency_diff.delta_ms:+.1f} ms` (`{self.latency_diff.percent_change:+.1f}%`)\n"
        )

        return "\n".join(md)

    def render_html(self) -> str:
        """Render HTML report with dark theme styling."""
        status_color = (
            "#3fb950"
            if self.semantic_diff.status == "MATCH"
            else "#d29922"
            if self.semantic_diff.status == "MINOR_DIFF"
            else "#f85149"
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sequa Phase 2 Regression Test Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    h1 {{ border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-top: 0; color: #f0f6fc; }}
    h2 {{ color: #58a6ff; margin-top: 24px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; color: #fff; background-color: {status_color}; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
    th {{ background: #21262d; color: #8b949e; }}
    pre {{ background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 6px; overflow-x: auto; font-family: monospace; font-size: 13px; }}
    .diff-add {{ color: #7ee787; background-color: #124027; }}
    .diff-sub {{ color: #ff7b72; background-color: #490202; }}
</style>
</head>
<body>
<div class="container">
    <h1>📼 Sequa Phase 2 Regression Test Report</h1>
    <p>Status: <span class="badge">{self.semantic_diff.status}</span></p>
    <table>
        <tr><th>Metric</th><th>Reference (Old)</th><th>New Execution</th><th>Delta / Status</th></tr>
        <tr><td>Cassette ID / Path</td><td><code>{self.old_id}</code></td><td><code>{self.new_id}</code></td><td>-</td></tr>
        <tr><td>Semantic Similarity</td><td>100%</td><td><code>{self.semantic_diff.similarity_score * 100:.1f}%</code></td><td><strong>{self.semantic_diff.status}</strong></td></tr>
        <tr><td>Total Tokens</td><td>{self.cost_diff.old_tokens['total']}</td><td>{self.cost_diff.new_tokens['total']}</td><td><code>{self.cost_diff.token_delta['total']:+} tokens</code></td></tr>
        <tr><td>Est. Cost (USD)</td><td>${self.cost_diff.old_cost_usd:.6f}</td><td>${self.cost_diff.new_cost_usd:.6f}</td><td><code>${self.cost_diff.cost_delta_usd:+.6f} ({self.cost_diff.cost_change_percent:+.1f}%)</code></td></tr>
        <tr><td>Latency</td><td>{self.latency_diff.old_latency_ms:.1f} ms</td><td>{self.latency_diff.new_latency_ms:.1f} ms</td><td><code>{self.latency_diff.delta_ms:+.1f} ms ({self.latency_diff.percent_change:+.1f}%)</code></td></tr>
    </table>

    <h2>1. 📝 Prompt Diff</h2>
    <pre>{self.prompt_diff.system_prompt_diff or self.prompt_diff.user_prompt_diff or "No prompt changes."}</pre>

    <h2>2. 🛠️ Tool Diff</h2>
    <pre>{json.dumps(self.tool_diff.to_dict(), indent=2)}</pre>

    <h2>3. 🧠 Semantic Output Diff</h2>
    <p>Similarity: <strong>{self.semantic_diff.similarity_score * 100:.1f}%</strong></p>
    <pre>{self.semantic_diff.diff_text or "No text output differences."}</pre>

    <h2>4. 💰 Cost & Token Usage</h2>
    <pre>{json.dumps(self.cost_diff.to_dict(), indent=2)}</pre>

    <h2>5. ⚡ Latency Performance</h2>
    <pre>{json.dumps(self.latency_diff.to_dict(), indent=2)}</pre>
</div>
</body>
</html>"""
        return html


def _to_data_dict(data_or_cassette: dict[str, Any] | Cassette | str) -> tuple[str, dict[str, Any]]:
    if isinstance(data_or_cassette, str):
        if data_or_cassette.endswith(".json") or "/" in data_or_cassette:
            cas = Cassette.load(data_or_cassette)
            return data_or_cassette, cas.to_dict()
        else:
            return data_or_cassette, {"id": data_or_cassette}
    elif isinstance(data_or_cassette, Cassette):
        return data_or_cassette.id or "cassette", data_or_cassette.to_dict()
    else:
        cid = data_or_cassette.get("id", "execution")
        return cid, data_or_cassette


def _extract_prompts_and_params(
    data: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    req = data.get("request", {})
    if not isinstance(req, dict):
        return "", "", {}

    system_prompt = ""
    user_prompts: list[str] = []

    messages = req.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                if role in ("system", "developer"):
                    system_prompt += f"{content}\n"
                elif role in ("user", "human"):
                    user_prompts.append(str(content))
    elif isinstance(req.get("prompt"), str):
        user_prompts.append(req["prompt"])

    params: dict[str, Any] = {}
    known_param_keys = [
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
    ]
    for key in known_param_keys:
        if key in req:
            params[key] = req[key]

    return system_prompt.strip(), "\n---\n".join(user_prompts).strip(), params


def _extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    req = data.get("request", {})
    res = data.get("response", {})
    metadata = data.get("metadata", {})

    tool_calls: list[dict[str, Any]] = []

    # Check response choices
    choices = res.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tcs = msg["tool_calls"]
                    if isinstance(tcs, list):
                        for tc in tcs:
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                                name = func.get("name") or tc.get("name", "unknown_tool")
                                args = func.get("arguments") or tc.get("arguments") or tc.get("args", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        pass
                                tool_calls.append({"name": name, "args": args})

    # Check Anthropic/LangChain content list
    content = res.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("tool_use", "tool_call"):
                    tool_calls.append({
                        "name": item.get("name", "unknown_tool"),
                        "args": item.get("input") or item.get("args", {}),
                    })

    # Check metadata or root response
    if not tool_calls:
        if "tool_calls" in metadata and isinstance(metadata["tool_calls"], list):
            for tc in metadata["tool_calls"]:
                if isinstance(tc, dict):
                    tool_calls.append({
                        "name": tc.get("name", "unknown_tool"),
                        "args": tc.get("args") or tc.get("arguments", {}),
                    })

    return tool_calls


def _extract_completion_text(data: dict[str, Any]) -> str:
    res = data.get("response", {})
    if isinstance(res.get("output"), str) and res["output"].strip():
        return res["output"].strip()

    choices = res.get("choices")
    if isinstance(choices, list):
        texts = []
        for choice in choices:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    texts.append(msg["content"])
                elif isinstance(choice.get("text"), str):
                    texts.append(choice["text"])
        if texts:
            return "\n".join(texts).strip()

    if isinstance(res.get("content"), str):
        return res["content"].strip()
    elif isinstance(res.get("content"), list):
        parts = []
        for p in res["content"]:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        if parts:
            return "\n".join(parts).strip()

    return ""


def _extract_token_usage(data: dict[str, Any]) -> dict[str, int]:
    res = data.get("response", {})
    meta = data.get("metadata", {})

    usage = res.get("usage") or meta.get("usage") or res.get("token_usage") or meta.get("token_usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or meta.get("prompt_tokens", 0)
    completion_tokens = (
        usage.get("completion_tokens") or usage.get("output_tokens") or meta.get("completion_tokens", 0)
    )
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    return {
        "prompt": int(prompt_tokens or 0),
        "completion": int(completion_tokens or 0),
        "total": int(total_tokens or 0),
    }


def _extract_latency(data: dict[str, Any]) -> float:
    meta = data.get("metadata", {})
    res = data.get("response", {})

    for key in ("duration_ms", "latency_ms", "elapsed_ms", "response_time_ms"):
        if key in meta and isinstance(meta[key], (int, float)):
            return float(meta[key])
        if key in res and isinstance(res[key], (int, float)):
            return float(res[key])
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])

    return 0.0


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Computes similarity score between 0.0 and 1.0 using token n-gram cosine similarity and SequenceMatcher."""
    s1, s2 = text1.strip(), text2.strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    def get_ngrams(text: str) -> Counter[str]:
        words = re.findall(r"\w+", text.lower())
        unigrams = words
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return Counter(unigrams + bigrams)

    c1 = get_ngrams(s1)
    c2 = get_ngrams(s2)

    intersection = set(c1.keys()) & set(c2.keys())
    dot_product = sum(c1[x] * c2[x] for x in intersection)
    mag1 = math.sqrt(sum(v**2 for v in c1.values()))
    mag2 = math.sqrt(sum(v**2 for v in c2.values()))

    if not mag1 or not mag2:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / (mag1 * mag2)

    seq_sim = difflib.SequenceMatcher(None, s1, s2).ratio()
    blend = (0.7 * cosine_sim) + (0.3 * seq_sim)
    return round(max(0.0, min(1.0, blend)), 4)


def estimate_cost_usd(model_name: str, tokens: dict[str, int]) -> float:
    model_name_lower = (model_name or "").lower()
    prompt_rate, completion_rate = DEFAULT_PRICING

    for k, rates in MODEL_PRICING.items():
        if k in model_name_lower:
            prompt_rate, completion_rate = rates
            break

    p_tokens = tokens.get("prompt", 0)
    c_tokens = tokens.get("completion", 0)

    cost = (p_tokens / 1000.0 * prompt_rate) + (c_tokens / 1000.0 * completion_rate)
    return cost


def compare_executions(
    old_cassette: Cassette | dict[str, Any] | str,
    new_execution: Cassette | dict[str, Any] | str,
) -> RegressionReport:
    """Compare an Old Cassette against a New Execution and return a 5-dimension RegressionReport."""
    old_id, data1 = _to_data_dict(old_cassette)
    new_id, data2 = _to_data_dict(new_execution)

    # 1. PROMPT DIFF
    old_sys, old_usr, old_params = _extract_prompts_and_params(data1)
    new_sys, new_usr, new_params = _extract_prompts_and_params(data2)

    param_changes: dict[str, dict[str, Any]] = {}
    all_keys = set(old_params.keys()) | set(new_params.keys())
    for k in all_keys:
        if old_params.get(k) != new_params.get(k):
            param_changes[k] = {"old": old_params.get(k), "new": new_params.get(k)}

    sys_diff = ""
    if old_sys != new_sys:
        sys_diff = "".join(
            difflib.unified_diff(
                old_sys.splitlines(keepends=True),
                new_sys.splitlines(keepends=True),
                fromfile="old_system_prompt",
                tofile="new_system_prompt",
            )
        )

    usr_diff = ""
    if old_usr != new_usr:
        usr_diff = "".join(
            difflib.unified_diff(
                old_usr.splitlines(keepends=True),
                new_usr.splitlines(keepends=True),
                fromfile="old_user_prompt",
                tofile="new_user_prompt",
            )
        )

    prompt_has_changes = bool(sys_diff or usr_diff or param_changes)
    prompt_diff_obj = PromptDiff(
        has_changes=prompt_has_changes,
        system_prompt_diff=sys_diff,
        user_prompt_diff=usr_diff,
        param_changes=param_changes,
        old_system_prompt=old_sys,
        new_system_prompt=new_sys,
        old_user_prompt=old_usr,
        new_user_prompt=new_usr,
    )

    # 2. TOOL DIFF
    old_tools = _extract_tool_calls(data1)
    new_tools = _extract_tool_calls(data2)

    added_tools = []
    removed_tools = []
    modified_tools = []

    # Map tools by name for matching
    old_tool_names = [t.get("name") for t in old_tools]
    new_tool_names = [t.get("name") for t in new_tools]

    for nt in new_tools:
        if nt.get("name") not in old_tool_names:
            added_tools.append(nt)

    for ot in old_tools:
        if ot.get("name") not in new_tool_names:
            removed_tools.append(ot)

    for ot in old_tools:
        matching_new = [nt for nt in new_tools if nt.get("name") == ot.get("name")]
        if matching_new:
            # Compare args
            if ot.get("args") != matching_new[0].get("args"):
                modified_tools.append({
                    "name": ot.get("name"),
                    "old_args": ot.get("args"),
                    "new_args": matching_new[0].get("args"),
                })

    order_changed = False
    if not added_tools and not removed_tools and len(old_tools) == len(new_tools):
        if [t.get("name") for t in old_tools] != [t.get("name") for t in new_tools]:
            order_changed = True

    tool_has_changes = bool(added_tools or removed_tools or modified_tools or order_changed)
    tool_diff_obj = ToolDiff(
        has_changes=tool_has_changes,
        added_tools=added_tools,
        removed_tools=removed_tools,
        modified_tools=modified_tools,
        order_changed=order_changed,
        old_tools=old_tools,
        new_tools=new_tools,
    )

    # 3. SEMANTIC DIFF
    old_text = _extract_completion_text(data1)
    new_text = _extract_completion_text(data2)

    sim_score = compute_semantic_similarity(old_text, new_text)
    status = "MATCH" if sim_score >= 0.95 else "MINOR_DIFF" if sim_score >= 0.80 else "SIGNIFICANT_DRIFT"

    text_diff = ""
    if old_text != new_text:
        text_diff = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile="old_response",
                tofile="new_response",
            )
        )

    semantic_diff_obj = SemanticDiff(
        similarity_score=sim_score,
        status=status,
        diff_text=text_diff,
        old_text=old_text,
        new_text=new_text,
    )

    # 4. COST DIFF
    old_tok = _extract_token_usage(data1)
    new_tok = _extract_token_usage(data2)

    tok_delta = {
        "prompt": new_tok["prompt"] - old_tok["prompt"],
        "completion": new_tok["completion"] - old_tok["completion"],
        "total": new_tok["total"] - old_tok["total"],
    }

    old_model = data1.get("request", {}).get("model") or data1.get("metadata", {}).get("model", "")
    new_model = data2.get("request", {}).get("model") or data2.get("metadata", {}).get("model", "")

    old_cost = estimate_cost_usd(str(old_model), old_tok)
    new_cost = estimate_cost_usd(str(new_model), new_tok)

    cost_delta = new_cost - old_cost
    pct_cost_change = (cost_delta / old_cost * 100.0) if old_cost > 0 else 0.0

    cost_diff_obj = CostDiff(
        old_tokens=old_tok,
        new_tokens=new_tok,
        token_delta=tok_delta,
        old_cost_usd=old_cost,
        new_cost_usd=new_cost,
        cost_delta_usd=cost_delta,
        cost_change_percent=pct_cost_change,
    )

    # 5. LATENCY DIFF
    old_lat = _extract_latency(data1)
    new_lat = _extract_latency(data2)

    lat_delta = new_lat - old_lat
    pct_lat_change = (lat_delta / old_lat * 100.0) if old_lat > 0 else 0.0

    latency_diff_obj = LatencyDiff(
        old_latency_ms=old_lat,
        new_latency_ms=new_lat,
        delta_ms=lat_delta,
        percent_change=pct_lat_change,
    )

    return RegressionReport(
        old_id=str(old_id),
        new_id=str(new_id),
        prompt_diff=prompt_diff_obj,
        tool_diff=tool_diff_obj,
        semantic_diff=semantic_diff_obj,
        cost_diff=cost_diff_obj,
        latency_diff=latency_diff_obj,
    )
