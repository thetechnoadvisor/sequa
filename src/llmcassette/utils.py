from __future__ import annotations

from typing import Any


def sort_dict_keys(data: Any) -> Any:
    """Recursively sort dictionary keys for deterministic hashing."""
    if isinstance(data, dict):
        return {k: sort_dict_keys(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        return [sort_dict_keys(item) for item in data]
    return data
