from __future__ import annotations

import json
import os
from typing import Any

from sequa.models import Cassette


def save(cassette: Cassette, path: str) -> None:
    """Save a cassette as a JSON file at the specified path."""
    # Ensure directory exists
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cassette.to_dict(), f, indent=4, ensure_ascii=False)


def load(path: str) -> Cassette:
    """Load a cassette from a JSON file at the specified path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Cassette.from_dict(data)


def exists(path: str) -> bool:
    """Check if a cassette file exists at the specified path."""
    return os.path.isfile(path)


def delete(path: str) -> None:
    """Delete the cassette file at the specified path if it exists."""
    if os.path.exists(path):
        os.remove(path)
