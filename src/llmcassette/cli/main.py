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
        try:
            with open(path, "r", encoding="utf-8") as f:
                cassettes.append((path, json.load(f)))
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Simple validation to verify it is an LLMCassette file
                            if "request" in data and "response" in data:
                                cassettes.append((file_path, data))
                    except Exception as e:
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
    print(" LLMCassette Statistics")
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
        from llmcassette.utils import sort_dict_keys
        data = sort_dict_keys(data)
        modified = True

        if modified:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                cleaned_count += 1
            except Exception as e:
                print(f"Error writing to {path}: {e}", file=sys.stderr)

    print(f"Successfully formatted/cleaned {cleaned_count} cassettes.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMCassette CLI - Manage and inspect your LLM snapshot cassettes."
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

    args = parser.parse_args()

    if args.command == "stats":
        sys.exit(cmd_stats(args))
    elif args.command == "inspect":
        sys.exit(cmd_inspect(args))
    elif args.command == "clean":
        sys.exit(cmd_clean(args))


if __name__ == "__main__":
    main()
