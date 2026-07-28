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
    storage: Any = field(default=None)

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

    def save(self, path_or_id: str | None = None, base_dir: str | None = None) -> None:
        from sequa.storage import resolve_storage
        st = resolve_storage(self.storage, base_dir=base_dir)
        st.save(self, cassette_id=path_or_id or self.id, base_dir=base_dir)

    @classmethod
    def load(cls, path_or_id: str, storage: Any = None) -> Cassette:
        from sequa.storage import resolve_storage
        st = resolve_storage(storage)
        return st.load(path_or_id)

    def __enter__(self) -> Any:
        from sequa.cassette import cassette
        self._cm = cassette(storage=self.storage)
        return self._cm.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self, "_cm"):
            return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, fn: Any) -> Any:
        from sequa.cassette import cassette
        return cassette(storage=self.storage)(fn)

