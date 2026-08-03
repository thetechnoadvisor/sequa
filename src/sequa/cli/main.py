from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import sys
from typing import Any


def load_all_cassettes(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Finds and loads all JSON cassette files in the given directory or file path."""
    cassettes = []
    if os.path.isfile(path) and path.endswith(".json"):
        if os.path.basename(path) == "metadata.json":
            return cassettes
        try:
            with open(path, "r", encoding="utf-8") as f:
                cassettes.append((path, json.load(f)))
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".json") and file != "metadata.json":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "request" in data and "response" in data:
                                cassettes.append((file_path, data))
                    except Exception:
                        pass
    return cassettes


def resolve_cassette(
    target: str, cassettes: list[tuple[str, dict[str, Any]]]
) -> tuple[str, dict[str, Any]] | None:
    """Helper to locate a cassette by file path, filename, hash, or ID."""
    target = target.strip()
    if os.path.isfile(target):
        try:
            with open(target, "r", encoding="utf-8") as f:
                return target, json.load(f)
        except Exception:
            pass

    for path, data in cassettes:
        cass_id = data.get("id", "")
        cass_hash = data.get("hash", "")
        filename = os.path.basename(path)

        if target in (cass_id, cass_hash, filename, path) or (
            len(target) >= 4 and (cass_hash.startswith(target) or cass_id.startswith(target))
        ):
            return path, data

    return None


def format_cassette_for_diff(
    data: dict[str, Any], ignore_fields: list[str] | None = None
) -> str:
    """Format canonical request, response, and metadata cleanly as JSON string for line-by-line diffing."""
    d = copy.deepcopy(data)
    if ignore_fields:
        for field in ignore_fields:
            d.pop(field, None)
            if "request" in d and isinstance(d["request"], dict):
                d["request"].pop(field, None)
            if "response" in d and isinstance(d["response"], dict):
                d["response"].pop(field, None)
            if "metadata" in d and isinstance(d["metadata"], dict):
                d["metadata"].pop(field, None)
    return json.dumps(d, indent=2, sort_keys=True, ensure_ascii=False)


def render_diff_text(
    label1: str, label2: str, json1_str: str, json2_str: str, use_color: bool = True
) -> str:
    """Render terminal text diff (with optional ANSI colors)."""
    lines1 = json1_str.splitlines(keepends=True)
    lines2 = json2_str.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2))

    if not diff:
        return "No differences found between cassettes.\n"

    output = []
    for line in diff:
        if use_color and sys.stdout.isatty():
            if line.startswith("---") or line.startswith("+++"):
                output.append(f"\033[1;36m{line}\033[0m")
            elif line.startswith("@@"):
                output.append(f"\033[33m{line}\033[0m")
            elif line.startswith("+"):
                output.append(f"\033[32m{line}\033[0m")
            elif line.startswith("-"):
                output.append(f"\033[31m{line}\033[0m")
            else:
                output.append(line)
        else:
            output.append(line)
    return "".join(output)


def render_diff_markdown(
    label1: str,
    label2: str,
    data1: dict[str, Any],
    data2: dict[str, Any],
    json1_str: str,
    json2_str: str,
) -> str:
    """Render clean GitHub Markdown comparison report with summary metadata table and diff fence."""
    lines1 = json1_str.splitlines(keepends=True)
    lines2 = json2_str.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2))

    md = []
    md.append("# Sequa Cassette Execution Diff\n\n")
    md.append(
        f"**Execution 1:** `{label1}` ({data1.get('provider', 'unknown')} / {data1.get('request', {}).get('model', 'unknown')})\n"
    )
    md.append(
        f"**Execution 2:** `{label2}` ({data2.get('provider', 'unknown')} / {data2.get('request', {}).get('model', 'unknown')})\n\n"
    )

    md.append("## Summary Comparison\n\n")
    md.append("| Metric | Execution 1 | Execution 2 |\n")
    md.append("| :--- | :--- | :--- |\n")
    md.append(f"| Hash | `{data1.get('hash', 'N/A')[:12]}` | `{data2.get('hash', 'N/A')[:12]}` |\n")
    md.append(f"| Provider | `{data1.get('provider', 'N/A')}` | `{data2.get('provider', 'N/A')}` |\n")
    md.append(
        f"| Model | `{data1.get('request', {}).get('model', 'N/A')}` | `{data2.get('request', {}).get('model', 'N/A')}` |\n"
    )
    md.append(
        f"| Created At | `{data1.get('created_at', 'N/A')}` | `{data2.get('created_at', 'N/A')}` |\n\n"
    )

    md.append("## Unified Diff\n\n")
    if not diff_lines:
        md.append("_No differences found between cassettes._\n")
    else:
        md.append("```diff\n")
        md.append("".join(diff_lines))
        md.append("```\n")

    return "".join(md)


def render_diff_html(
    label1: str,
    label2: str,
    data1: dict[str, Any],
    data2: dict[str, Any],
    json1_str: str,
    json2_str: str,
) -> str:
    """Render HTML diff report with modern dark theme styling."""
    lines1 = json1_str.splitlines()
    lines2 = json2_str.splitlines()

    html_diff = difflib.HtmlDiff().make_table(
        lines1, lines2, fromdesc=label1, todesc=label2, context=True, numlines=3
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sequa Cassette Diff: {label1} vs {label2}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 20px; }}
    h1 {{ color: #58a6ff; font-size: 24px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
    .meta-table {{ border-collapse: collapse; margin-bottom: 20px; width: 100%; max-width: 800px; }}
    .meta-table th, .meta-table td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
    .meta-table th {{ background-color: #161b22; color: #8b949e; }}
    table.diff {{ font-family: monospace; font-size: 13px; border-collapse: collapse; width: 100%; background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
    .diff td, .diff th {{ padding: 2px 6px; border: none; }}
    .diff_header {{ background-color: #21262d; color: #8b949e; text-align: right; width: 40px; user-select: none; }}
    .diff_next {{ background-color: #21262d; width: 20px; text-align: center; }}
    .diff_add {{ background-color: #124027; color: #7ee787; }}
    .diff_chg {{ background-color: #3b2300; color: #d29922; }}
    .diff_sub {{ background-color: #490202; color: #ff7b72; }}
</style>
</head>
<body>
<h1>📼 Sequa Cassette Execution Diff</h1>
<table class="meta-table">
    <tr><th>Metric</th><th>Execution 1 ({label1})</th><th>Execution 2 ({label2})</th></tr>
    <tr><td>Hash</td><td><code>{data1.get('hash', 'N/A')[:12]}</code></td><td><code>{data2.get('hash', 'N/A')[:12]}</code></td></tr>
    <tr><td>Provider</td><td><code>{data1.get('provider', 'N/A')}</code></td><td><code>{data2.get('provider', 'N/A')}</code></td></tr>
    <tr><td>Model</td><td><code>{data1.get('request', {}).get('model', 'N/A')}</code></td><td><code>{data2.get('request', {}).get('model', 'N/A')}</code></td></tr>
    <tr><td>Created At</td><td><code>{data1.get('created_at', 'N/A')}</code></td><td><code>{data2.get('created_at', 'N/A')}</code></td></tr>
</table>
<h2>Differences Table</h2>
{html_diff}
</body>
</html>
"""


def cmd_stats(args: argparse.Namespace) -> int:
    """Calculates and prints statistics for stored cassettes."""
    cassettes = load_all_cassettes(args.path)
    if not cassettes:
        print(f"No cassettes found at path: {args.path}")
        return 0

    count = len(cassettes)
    total_latency_ms = 0.0
    total_size_bytes = 0

    for path, data in cassettes:
        try:
            total_size_bytes += os.path.getsize(path)
        except Exception:
            pass

        latency = data.get("metadata", {}).get("latency_ms")
        if latency is None:
            latency = data.get("response", {}).get("latency")

        if latency is not None:
            try:
                total_latency_ms += float(latency)
            except ValueError:
                pass

    print("========================================")
    print(" Sequa Statistics")
    print("========================================")
    print(f"Total Cassettes:      {count}")
    print(f"Total Size on Disk:   {total_size_bytes / 1024:.2f} KB ({total_size_bytes} bytes)")
    print(f"Total Latency Saved:  {total_latency_ms / 1000:.2f} seconds ({total_latency_ms:.1f} ms)")
    print("========================================")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Lists summary of all stored cassettes."""
    cassettes = load_all_cassettes(args.path)
    if not cassettes:
        print(f"No cassettes found at path: {args.path}")
        return 0

    print(f"{'Filename / Hash':<40} | {'Provider':<15} | {'Model':<25} | {'Created At':<25}")
    print("-" * 111)
    for path, data in cassettes:
        filename = os.path.basename(path)
        provider = data.get("provider", "unknown")

        req = data.get("request", {})
        model = req.get("model") or "unknown"

        created_at = data.get("created_at", "unknown")

        if len(filename) > 38:
            filename_display = filename[:35] + "..."
        else:
            filename_display = filename

        print(f"{filename_display:<40} | {provider:<15} | {model:<25} | {created_at:<25}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Cleans dynamic/volatile fields from cassettes or formats them."""
    cassettes = load_all_cassettes(args.path)
    if not cassettes:
        print(f"No cassettes found at path: {args.path}")
        return 0

    cleaned_count = 0
    for path, data in cassettes:
        modified = False

        if args.remove_latency:
            if "metadata" in data and "latency_ms" in data["metadata"]:
                del data["metadata"]["latency_ms"]
                modified = True
            if "response" in data and "latency" in data["response"]:
                del data["response"]["latency"]
                modified = True

        if args.remove_timestamps:
            if "created_at" in data:
                data["created_at"] = ""
                modified = True

        from sequa.utils import sort_dict_keys

        data = sort_dict_keys(data)
        modified = True

        if modified:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                cleaned_count += 1
            except Exception as e:
                print(f"Error writing to {path}: {e}", file=sys.stderr)

    if os.path.isdir(args.path):
        from sequa.storage import update_metadata_index

        update_metadata_index(args.path)

    print(f"Successfully formatted/cleaned {cleaned_count} cassettes.")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    """Git-like log listing execution cassettes chronologically."""
    from sequa.search import extract_searchable_text

    cassettes = load_all_cassettes(args.path)
    if not cassettes:
        print(f"No cassettes found at path: {args.path}")
        return 0

    def sort_key(item: tuple[str, dict[str, Any]]) -> str:
        return item[1].get("created_at") or ""

    sorted_cassettes = sorted(cassettes, key=sort_key, reverse=True)
    limit = getattr(args, "number", None) or 10
    display_list = sorted_cassettes[:limit]

    print("================================================================================")
    print(
        f" Sequa Cassette Execution Log (Showing {len(display_list)} of {len(sorted_cassettes)})"
    )
    print("================================================================================")

    for path, data in display_list:
        cass_hash = data.get("hash") or os.path.basename(path).replace(".json", "")
        short_hash = cass_hash[:12] if len(cass_hash) > 12 else cass_hash
        provider = data.get("provider", "unknown")
        model = data.get("request", {}).get("model") or "unknown"
        created_at = data.get("created_at", "unknown")

        latency = data.get("metadata", {}).get("latency_ms")
        if latency is None:
            latency = data.get("response", {}).get("latency")
        latency_str = f"{float(latency):.1f} ms" if latency is not None else "N/A"

        _, in_snip, out_snip = extract_searchable_text(data)

        print(f"cassette {short_hash} ({provider} / {model})")
        print(f"Date:     {created_at}")
        print(f"Latency:  {latency_str}")
        print(f"Path:     {path}")
        print(f"  Input:  {in_snip}")
        print(f"  Output: {out_snip}")
        print("-" * 80)

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search cassettes with cosine similarity and time/metadata filters."""
    from sequa.search import search_cassettes

    results = search_cassettes(
        query=args.query or "",
        path=args.path,
        since=args.since,
        until=args.until,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
    )

    if not results:
        print("No matching cassettes found.")
        return 0

    print("================================================================================")
    print(f" Search Results for: '{args.query or '*'}' (Found {len(results)})")
    print("================================================================================")
    print(f"{'Idx':<4} | {'Score':<7} | {'Hash':<12} | {'Provider':<15} | {'Model':<20}")
    print("-" * 80)

    for idx, res in enumerate(results, 1):
        short_hash = res.hash[:12] if len(res.hash) > 12 else res.hash
        score_str = f"{res.score:.4f}"
        print(
            f"{idx:<4} | {score_str:<7} | {short_hash:<12} | {res.provider[:15]:<15} | {res.model[:20]:<20}"
        )
        print(f"     Input:  {res.input_snippet}")
        print(f"     Output: {res.output_snippet}")
        print("-" * 80)

    if getattr(args, "interactive", False):
        try:
            user_input = input(
                "\nEnter cassette number to inspect (e.g. 1) or diff two results (e.g. 1,2 or diff 1 2) [Enter to exit]: "
            ).strip()
            if not user_input:
                return 0

            cleaned_input = user_input.replace("diff", "").replace(",", " ").strip()
            parts = cleaned_input.split()

            if len(parts) == 2 and all(p.isdigit() for p in parts):
                idx1, idx2 = int(parts[0]), int(parts[1])
                if 1 <= idx1 <= len(results) and 1 <= idx2 <= len(results):
                    res1 = results[idx1 - 1]
                    res2 = results[idx2 - 1]
                    print("\n" + "=" * 80)
                    print(
                        f" Interactive Diff: Result #{idx1} ({res1.hash[:12]}) vs Result #{idx2} ({res2.hash[:12]})"
                    )
                    print("=" * 80)

                    diff_args = argparse.Namespace(
                        execution_1=res1.file_path,
                        execution_2=res2.file_path,
                        path=args.path,
                        format="text",
                        output=None,
                        ignore_fields=[],
                    )
                    cmd_diff(diff_args)
            elif user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(results):
                    sel = results[choice - 1]
                    print("\n" + "=" * 60)
                    print(f" Selected Cassette: {sel.hash}")
                    print("=" * 60)
                    print(f"File Path: {sel.file_path}")
                    print(f"Created At: {sel.created_at}")
                    print("\n--- Python Replay Code Snippet ---")
                    cassette_dir = os.path.dirname(sel.file_path)
                    print("from sequa import cassette\n")
                    print(f'with cassette("{cassette_dir}"):')
                    print("    # Run your AI execution here - response will be replayed!")
                    print("=" * 60)
        except (KeyboardInterrupt, EOFError):
            pass

    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Inspect and generate code snippet to replay a specific cassette."""
    cassettes = load_all_cassettes(args.path)
    res = resolve_cassette(args.target, cassettes)

    if not res:
        print(
            f"Error: Cassette matching '{args.target}' not found at path: {args.path}",
            file=sys.stderr,
        )
        return 1

    selected_path, selected_data = res

    from sequa.search import extract_searchable_text

    _, in_snip, out_snip = extract_searchable_text(selected_data)

    print("================================================================================")
    print(f" Sequa Replay Target: {selected_data.get('hash', args.target)}")
    print("================================================================================")
    print(f"File Path:  {selected_path}")
    print(f"Provider:   {selected_data.get('provider', 'unknown')}")
    print(f"Model:      {selected_data.get('request', {}).get('model', 'unknown')}")
    print(f"Created At: {selected_data.get('created_at', 'unknown')}")
    print(f"Input:      {in_snip}")
    print(f"Output:     {out_snip}")
    print("\n--------------------------------------------------------------------------------")
    print(" Replay Snippet (copy into your test suite / code):")
    print("--------------------------------------------------------------------------------")
    dir_path = os.path.dirname(selected_path)
    print("from sequa import cassette\n")
    print(f'with cassette("{dir_path}"):')
    print("    # Execute model call here; response will be replayed deterministically!")
    print("================================================================================")

    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two cassette recordings and display/save diff in text, markdown, or HTML format."""
    cassettes = load_all_cassettes(args.path)

    res1 = resolve_cassette(args.execution_1, cassettes)
    res2 = resolve_cassette(args.execution_2, cassettes)

    if not res1:
        print(
            f"Error: Execution 1 '{args.execution_1}' not found at path: {args.path}",
            file=sys.stderr,
        )
        return 1
    if not res2:
        print(
            f"Error: Execution 2 '{args.execution_2}' not found at path: {args.path}",
            file=sys.stderr,
        )
        return 1

    path1, data1 = res1
    path2, data2 = res2

    label1 = data1.get("hash", "")[:12] or os.path.basename(path1)
    label2 = data2.get("hash", "")[:12] or os.path.basename(path2)

    ignore_fields = getattr(args, "ignore_fields", None) or []
    json1_str = format_cassette_for_diff(data1, ignore_fields)
    json2_str = format_cassette_for_diff(data2, ignore_fields)

    fmt = (getattr(args, "format", None) or "text").lower()

    if fmt == "html" or (args.output and args.output.endswith(".html")):
        diff_output = render_diff_html(label1, label2, data1, data2, json1_str, json2_str)
    elif fmt in ("markdown", "md") or (args.output and args.output.endswith(".md")):
        diff_output = render_diff_markdown(label1, label2, data1, data2, json1_str, json2_str)
    else:
        diff_output = render_diff_text(label1, label2, json1_str, json2_str)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(diff_output)
            print(f"Diff report successfully saved to: {args.output}")
        except Exception as e:
            print(f"Error writing output file {args.output}: {e}", file=sys.stderr)
            return 1
    else:
        print(diff_output)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequa CLI - Manage and inspect your LLM snapshot cassettes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats
    parser_stats = subparsers.add_parser("stats", help="Show summary statistics of stored cassettes.")
    parser_stats.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory or file (default: 'cassettes').",
    )

    # inspect
    parser_inspect = subparsers.add_parser("inspect", help="List summary of all stored cassettes.")
    parser_inspect.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory or file (default: 'cassettes').",
    )

    # clean
    parser_clean = subparsers.add_parser("clean", help="Clean or format stored cassettes.")
    parser_clean.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory or file (default: 'cassettes').",
    )
    parser_clean.add_argument(
        "--remove-latency",
        action="store_true",
        help="Remove dynamic latency values from cassettes to avoid git diff noise.",
    )
    parser_clean.add_argument(
        "--remove-timestamps",
        action="store_true",
        help="Redact/clear dynamic timestamps.",
    )

    # log
    parser_log = subparsers.add_parser("log", help="Git-like log listing execution cassettes chronologically.")
    parser_log.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory (default: 'cassettes').",
    )
    parser_log.add_argument(
        "-n",
        "--number",
        type=int,
        default=10,
        help="Maximum number of log entries to display.",
    )

    # search
    parser_search = subparsers.add_parser("search", help="Search cassettes with cosine similarity and time filters.")
    parser_search.add_argument(
        "query",
        nargs="?",
        default="",
        help="Text query to search for using cosine similarity.",
    )
    parser_search.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory (default: 'cassettes').",
    )
    parser_search.add_argument(
        "--since",
        "-s",
        help="Filter cassettes created since relative time (e.g. '10m', '2h', '1d') or ISO date.",
    )
    parser_search.add_argument(
        "--until",
        "-u",
        help="Filter cassettes created until relative time or ISO date.",
    )
    parser_search.add_argument(
        "--provider",
        help="Filter by provider name (e.g. 'openai', 'groq', 'langchain_groq').",
    )
    parser_search.add_argument(
        "--model",
        "-m",
        help="Filter by model name.",
    )
    parser_search.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top search results to return.",
    )
    parser_search.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively select a search result to view details, replay snippet, or diff two results.",
    )

    # replay
    parser_replay = subparsers.add_parser("replay", help="Inspect and get replay snippet for a cassette.")
    parser_replay.add_argument(
        "target",
        help="Hash, ID, or file path of the cassette to replay.",
    )
    parser_replay.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory (default: 'cassettes').",
    )

    # diff
    parser_diff = subparsers.add_parser(
        "diff", help="Compare two cassette recordings and display/save diff."
    )
    parser_diff.add_argument(
        "execution_1",
        help="Hash, ID, or file path of the first cassette execution.",
    )
    parser_diff.add_argument(
        "execution_2",
        help="Hash, ID, or file path of the second cassette execution.",
    )
    parser_diff.add_argument(
        "--path",
        "-p",
        default="cassettes",
        help="Path to the cassettes directory (default: 'cassettes').",
    )
    parser_diff.add_argument(
        "--format",
        "-f",
        choices=["text", "markdown", "md", "html"],
        default="text",
        help="Output format: text (default), markdown, or html.",
    )
    parser_diff.add_argument(
        "--output",
        "-o",
        help="File path to save the diff output (e.g. diff.html or diff.md).",
    )
    parser_diff.add_argument(
        "--ignore-fields",
        nargs="*",
        default=[],
        help="Additional fields to exclude when computing diff.",
    )

    args = parser.parse_args()

    if args.command == "stats":
        sys.exit(cmd_stats(args))
    elif args.command == "inspect":
        sys.exit(cmd_inspect(args))
    elif args.command == "clean":
        sys.exit(cmd_clean(args))
    elif args.command == "log":
        sys.exit(cmd_log(args))
    elif args.command == "search":
        sys.exit(cmd_search(args))
    elif args.command == "replay":
        sys.exit(cmd_replay(args))
    elif args.command == "diff":
        sys.exit(cmd_diff(args))


if __name__ == "__main__":
    main()
