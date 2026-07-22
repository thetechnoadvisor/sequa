from __future__ import annotations

from typing import Any


def sort_dict_keys(data: Any) -> Any:
    """Recursively sort dictionary keys for deterministic hashing."""
    if isinstance(data, dict):
        return {k: sort_dict_keys(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        return [sort_dict_keys(item) for item in data]
    return data


def serialize_type_or_pydantic(val: Any) -> Any:
    """Recursively convert Python types, Pydantic model classes, dataclasses, and functions into JSON-serializable representations."""
    if isinstance(val, type):
        if hasattr(val, "model_json_schema"):
            try:
                return {
                    "__type__": "pydantic_model",
                    "name": val.__name__,
                    "schema": val.model_json_schema(),
                }
            except Exception:
                pass
        if hasattr(val, "schema"):
            try:
                return {
                    "__type__": "pydantic_model",
                    "name": val.__name__,
                    "schema": val.schema(),
                }
            except Exception:
                pass
        return {
            "__type__": "type",
            "name": getattr(val, "__name__", str(val)),
            "module": getattr(val, "__module__", ""),
        }
    if callable(val) and not isinstance(val, (dict, list, tuple)):
        name = getattr(val, "__name__", str(val))
        mod = getattr(val, "__module__", "")
        return {"__type__": "function", "name": name, "module": mod}
    if isinstance(val, dict):
        return {k: serialize_type_or_pydantic(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize_type_or_pydantic(v) for v in val]
    if isinstance(val, tuple):
        return tuple(serialize_type_or_pydantic(v) for v in val)
    if isinstance(val, set):
        return {serialize_type_or_pydantic(v) for v in val}
    return val

