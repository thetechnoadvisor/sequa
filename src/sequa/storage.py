from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

from sequa.models import Cassette


class StorageBackend(ABC):
    """Abstract base class for cassette storage backends."""

    @abstractmethod
    def save(self, cassette: Cassette | dict[str, Any], cassette_id: str | None = None, **kwargs: Any) -> None:
        """Save a Cassette object to the storage space."""
        pass

    @abstractmethod
    def load(self, cassette_id: str) -> Cassette:
        """Load a Cassette object by ID or path from the storage space."""
        pass

    @abstractmethod
    def exists(self, cassette_id: str) -> bool:
        """Check if a cassette exists in the storage space."""
        pass

    @abstractmethod
    def delete(self, cassette_id: str) -> None:
        """Delete a cassette from the storage space."""
        pass

    @abstractmethod
    def list(self) -> list[str]:
        """List all cassette IDs or keys in the storage space."""
        pass


class FileStorage(StorageBackend):
    """File-based cassette storage backend."""

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = base_dir

    def save(self, cassette: Cassette | dict[str, Any], cassette_id: str | None = None, base_dir: str | None = None, **kwargs: Any) -> None:
        if isinstance(cassette, dict):
            cassette_obj = Cassette.from_dict(cassette)
        else:
            cassette_obj = cassette

        target_path = cassette_id or kwargs.get("path")
        if not target_path:
            target_path = f"{cassette_obj.id}.json"
            if self.base_dir:
                target_path = os.path.join(self.base_dir, target_path)

        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(cassette_obj.to_dict(), f, indent=4, ensure_ascii=False)

        target_base = _get_base_dir(target_path, base_dir or self.base_dir)
        if target_base:
            update_metadata_index(target_base)

    def load(self, cassette_id: str) -> Cassette:
        if not os.path.isfile(cassette_id):
            raise FileNotFoundError(f"Cassette file not found: {cassette_id}")
        with open(cassette_id, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Cassette.from_dict(data)

    def exists(self, cassette_id: str) -> bool:
        return os.path.isfile(cassette_id)

    def delete(self, cassette_id: str, base_dir: str | None = None, **kwargs: Any) -> None:
        if os.path.exists(cassette_id):
            os.remove(cassette_id)
            target_base = _get_base_dir(cassette_id, base_dir or self.base_dir)
            if target_base:
                update_metadata_index(target_base)

    def list(self) -> list[str]:
        root_dir = self.base_dir or "."
        cassette_files: list[str] = []
        if os.path.isdir(root_dir):
            for root, _, files in os.walk(root_dir):
                for file in sorted(files):
                    if file.endswith(".json") and file != "metadata.json":
                        cassette_files.append(os.path.join(root, file))
        return cassette_files


class MemoryStorage(StorageBackend):
    """In-memory cassette storage backend."""

    def __init__(self) -> None:
        self._store: dict[str, Cassette] = {}

    def save(self, cassette: Cassette | dict[str, Any], cassette_id: str | None = None, **kwargs: Any) -> None:
        if isinstance(cassette, dict):
            cassette_obj = Cassette.from_dict(cassette)
        else:
            cassette_obj = cassette

        key = cassette_id or getattr(cassette_obj, "id", None) or getattr(cassette_obj, "hash", None)
        if not key:
            key = str(uuid.uuid4())

        self._store[key] = cassette_obj

    def _resolve_key(self, cassette_id: str) -> str | None:
        if cassette_id in self._store:
            return cassette_id

        norm_target = cassette_id.replace("\\", "/").strip("/")
        base_target = os.path.basename(cassette_id)
        base_target_no_ext = base_target[:-5] if base_target.endswith(".json") else base_target

        for stored_key, cassette_obj in list(self._store.items()):
            norm_stored = stored_key.replace("\\", "/").strip("/")
            stored_base = os.path.basename(stored_key)

            if norm_target == norm_stored or base_target == stored_base:
                return stored_key
            if cassette_obj.id == cassette_id or cassette_obj.hash == cassette_id:
                return stored_key
            if cassette_obj.hash and (cassette_obj.hash == base_target_no_ext or cassette_obj.hash in cassette_id):
                return stored_key

        return None

    def exists(self, cassette_id: str) -> bool:
        return self._resolve_key(cassette_id) is not None

    def load(self, cassette_id: str) -> Cassette:
        resolved = self._resolve_key(cassette_id)
        if resolved is None:
            raise FileNotFoundError(f"Cassette '{cassette_id}' not found in MemoryStorage.")
        return self._store[resolved]

    def delete(self, cassette_id: str) -> None:
        resolved = self._resolve_key(cassette_id)
        if resolved and resolved in self._store:
            del self._store[resolved]

    def list(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all stored cassettes in memory."""
        self._store.clear()


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


def resolve_storage(storage: StorageBackend | str | None = None, base_dir: str | None = None) -> StorageBackend:
    """Resolve a storage argument (string option like 'memory'/'file', StorageBackend instance, or None)."""
    if storage is None:
        return FileStorage(base_dir=base_dir)
    if isinstance(storage, StorageBackend):
        return storage
    if isinstance(storage, str):
        st_lower = storage.strip().lower()
        if st_lower in ("memory", "ram", "in_memory"):
            return MemoryStorage()
        elif st_lower in ("file", "disk", "local"):
            return FileStorage(base_dir=base_dir)
        else:
            raise ValueError(
                f"Invalid storage option: '{storage}'. Choose from 'file', 'memory', or pass a StorageBackend instance."
            )
    raise TypeError(f"Invalid storage type: {type(storage)}. Expected str, StorageBackend instance, or None.")


_default_file_storage = FileStorage()


def save(cassette: Cassette, path: str, base_dir: str | None = None) -> None:
    """Save a cassette as a JSON file at the specified path."""
    _default_file_storage.save(cassette, cassette_id=path, base_dir=base_dir)


def load(path: str) -> Cassette:
    """Load a cassette from a JSON file at the specified path."""
    return _default_file_storage.load(path)


def exists(path: str) -> bool:
    """Check if a cassette file exists at the specified path."""
    return _default_file_storage.exists(path)


def delete(path: str, base_dir: str | None = None) -> None:
    """Delete the cassette file at the specified path if it exists."""
    _default_file_storage.delete(path, base_dir=base_dir)



