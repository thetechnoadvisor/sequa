from __future__ import annotations

import argparse
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
                            # Simple validation to verify it is a Sequa cassette file
                            if "request" in data and "response" in data:
                                cassettes.append((file_path, data))
                    except Exception:
                        # Skip files that are not valid JSON or not cassettes
                        pass
    return cassettes


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
        # Size
        try:
            total_size_bytes += os.path.getsize(path)
        except Exception:
            pass
        
        # Latency
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
        
        # Get model
        req = data.get("request", {})
        model = req.get("model") or "unknown"
        
        created_at = data.get("created_at", "unknown")
        
        # Format filename to fit table neatly
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
        
        # 1. Remove latency metadata if requested
        if args.remove_latency:
            if "metadata" in data and "latency_ms" in data["metadata"]:
                del data["metadata"]["latency_ms"]
                modified = True
            if "response" in data and "latency" in data["response"]:
                del data["response"]["latency"]
                modified = True

        # 2. Remove timestamps if requested
        if args.remove_timestamps:
            if "created_at" in data:
                data["created_at"] = ""
                modified = True

        # 3. Format/Sort keys cleanly (always done during clean)
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

    # Sort descending by created_at
    def sort_key(item: tuple[str, dict[str, Any]]) -> str:
        return item[1].get("created_at") or ""

    sorted_cassettes = sorted(cassettes, key=sort_key, reverse=True)
    limit = getattr(args, "number", None) or 10
    display_list = sorted_cassettes[:limit]

    print("================================================================================")
    print(f" Sequa Cassette Execution Log (Showing {len(display_list)} of {len(sorted_cassettes)})")
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
        print(f"{idx:<4} | {score_str:<7} | {short_hash:<12} | {res.provider[:15]:<15} | {res.model[:20]:<20}")
        print(f"     Input:  {res.input_snippet}")
        print(f"     Output: {res.output_snippet}")
        print("-" * 80)

    if getattr(args, "interactive", False):
        try:
            user_input = input("\nEnter cassette number to inspect & replay (or press Enter to exit): ").strip()
            if user_input.isdigit():
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
    target = args.target.strip()

    selected_path = None
    selected_data = None

    for path, data in cassettes:
        cass_id = data.get("id", "")
        cass_hash = data.get("hash", "")
        filename = os.path.basename(path)

        if target in (cass_id, cass_hash, filename, path) or (len(target) >= 6 and (cass_hash.startswith(target) or cass_id.startswith(target))):
            selected_path = path
            selected_data = data
            break

    if not selected_path or not selected_data:
        print(f"Error: Cassette matching '{target}' not found at path: {args.path}", file=sys.stderr)
        return 1

    from sequa.search import extract_searchable_text
    _, in_snip, out_snip = extract_searchable_text(selected_data)

    print("================================================================================")
    print(f" Sequa Replay Target: {selected_data.get('hash', target)}")
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
        help="Interactively select a search result to view details and replay snippet.",
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


if __name__ == "__main__":
    main()

