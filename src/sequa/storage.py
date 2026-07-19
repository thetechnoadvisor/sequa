from __future__ import annotations

import json
import os
from typing import Any

from sequa.models import Cassette


def update_metadata_index(base_dir: str) -> None:
    """Generate or update metadata.json in base_dir listing all stored cassettes."""
    if not os.path.isdir(base_dir):
        return

    cassettes_info: dict[str, dict[str, Any]] = {}
    providers_summary: dict[str, int] = {}

    for root, _, files in os.walk(base_dir):
        for file in sorted(files):
            if file.endswith(".json") and file != "metadata.json":
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if not isinstance(data, dict) or "request" not in data or "response" not in data:
                        continue

                    provider = data.get("provider", "")
                    if not provider:
                        rel_parts = rel_path.split(os.sep)
                        if len(rel_parts) > 1:
                            provider = rel_parts[0]
                        else:
                            provider = "unknown"

                    req = data.get("request", {})
                    model = req.get("model") or ""
                    created_at = data.get("created_at") or ""
                    cassette_id = data.get("id") or ""
                    cassette_hash = data.get("hash") or os.path.splitext(file)[0]
                    latency = data.get("metadata", {}).get("latency_ms")
                    if latency is None:
                        latency = data.get("response", {}).get("latency")

                    entry: dict[str, Any] = {
                        "id": cassette_id,
                        "hash": cassette_hash,
                        "provider": provider,
                        "model": model,
                        "file": rel_path.replace("\\", "/"),
                        "created_at": created_at,
                    }
                    if latency is not None:
                        entry["latency_ms"] = latency

                    key = cassette_hash if cassette_hash else rel_path.replace("\\", "/")
                    cassettes_info[key] = entry

                    providers_summary[provider] = providers_summary.get(provider, 0) + 1
                except Exception:
                    pass

    metadata_content = {
        "version": "1.0",
        "total_cassettes": len(cassettes_info),
        "providers": providers_summary,
        "cassettes": cassettes_info,
    }

    metadata_path = os.path.join(base_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=4, ensure_ascii=False)


def _get_base_dir(path: str, base_dir: str | None = None) -> str | None:
    if base_dir:
        return base_dir if os.path.isdir(base_dir) else os.path.dirname(os.path.abspath(base_dir))
    
    abs_path = os.path.abspath(path)
    parent_dir = os.path.dirname(abs_path)
    if not parent_dir:
        return None
    
    grandparent_dir = os.path.dirname(parent_dir)
    if grandparent_dir and grandparent_dir != parent_dir:
        folder_name = os.path.basename(parent_dir).lower()
        if os.path.exists(os.path.join(grandparent_dir, "metadata.json")) or folder_name in (
            "anthropic", "openai", "groq", "langchain_groq", "default", "unknown"
        ):
            return grandparent_dir
            
    return parent_dir


def save(cassette: Cassette, path: str, base_dir: str | None = None) -> None:
    """Save a cassette as a JSON file at the specified path."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cassette.to_dict(), f, indent=4, ensure_ascii=False)

    target_base = _get_base_dir(path, base_dir)
    if target_base:
        update_metadata_index(target_base)


def load(path: str) -> Cassette:
    """Load a cassette from a JSON file at the specified path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Cassette.from_dict(data)


def exists(path: str) -> bool:
    """Check if a cassette file exists at the specified path."""
    return os.path.isfile(path)


def delete(path: str, base_dir: str | None = None) -> None:
    """Delete the cassette file at the specified path if it exists."""
    if os.path.exists(path):
        os.remove(path)
        target_base = _get_base_dir(path, base_dir)
        if target_base:
            update_metadata_index(target_base)

