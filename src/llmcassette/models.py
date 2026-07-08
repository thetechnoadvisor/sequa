from __future__ import annotations

import datetime
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Cassette:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = ""
    hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "hash": self.hash,
            "created_at": self.created_at,
            "request": self.request,
            "response": self.response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            provider=data.get("provider", ""),
            hash=data.get("hash", ""),
            created_at=data.get("created_at", ""),
            request=data.get("request", {}),
            response=data.get("response", {}),
            metadata=data.get("metadata", {}),
        )
