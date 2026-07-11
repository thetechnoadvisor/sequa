from __future__ import annotations

import hashlib
import json
from typing import Any

from sequa.llm.adapters.base import CanonicalRequest
from sequa.utils import sort_dict_keys


def normalize(req: dict[str, Any] | CanonicalRequest) -> dict[str, Any]:
    """Normalize request into a standard dictionary structure."""
    if isinstance(req, CanonicalRequest):
        req_dict = {
            "provider": req.provider,
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "params": req.params,
        }
    else:
        req_dict = dict(req)

    # Normalize messages list to standard role/content/type structure
    messages = req_dict.get("messages")
    if messages is not None:
        normalized_msgs = []
        for msg in messages:
            if isinstance(msg, dict):
                # Keep only core message fields
                normalized_msgs.append(
                    {k: v for k, v in msg.items() if k in ("role", "content", "type", "name")}
                )
            elif hasattr(msg, "to_dict"):
                try:
                    d = msg.to_dict()
                    normalized_msgs.append(
                        {k: v for k, v in d.items() if k in ("role", "content", "type", "name")}
                    )
                except Exception:
                    normalized_msgs.append({"content": str(msg)})
            elif hasattr(msg, "dict"):
                try:
                    d = msg.dict()
                    normalized_msgs.append(
                        {k: v for k, v in d.items() if k in ("role", "content", "type", "name")}
                    )
                except Exception:
                    normalized_msgs.append({"content": str(msg)})
            else:
                normalized_msgs.append({"content": str(msg)})
        req_dict["messages"] = normalized_msgs

    # Standardize types and remove None values to ensure consistent hashing
    cleaned: dict[str, Any] = {}
    for k, v in req_dict.items():
        if v is not None:
            if k == "params" and isinstance(v, dict):
                # Deep clean params
                cleaned_params = {pk: pv for pk, pv in v.items() if pv is not None}
                cleaned[k] = cleaned_params
            else:
                cleaned[k] = v

    return cleaned


def hash_request(
    req: dict[str, Any] | CanonicalRequest, ignore_fields: list[str] | None = None
) -> str:
    """Normalize, strip ignored fields, recursively sort, serialize, and hash a request."""
    # 1. Normalize
    normalized = normalize(req)

    # 2. Remove ignored fields
    ignore = ignore_fields or []
    for field_name in ignore:
        if field_name in normalized:
            del normalized[field_name]
        if "params" in normalized and isinstance(normalized["params"], dict):
            if field_name in normalized["params"]:
                # Copy to avoid modifying the input request dictionary in-place
                normalized["params"] = dict(normalized["params"])
                del normalized["params"][field_name]

    # 3. Recursive sort
    sorted_norm = sort_dict_keys(normalized)

    # 4. Serialize with compact JSON notation
    serialized = json.dumps(
        sorted_norm, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )

    # 5. SHA-256 Hash
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def match(
    req1: dict[str, Any] | CanonicalRequest,
    req2: dict[str, Any] | CanonicalRequest,
    ignore_fields: list[str] | None = None,
) -> bool:
    """Compare two requests to check if their hashes match after normalization and ignore filtering."""
    return hash_request(req1, ignore_fields) == hash_request(req2, ignore_fields)
