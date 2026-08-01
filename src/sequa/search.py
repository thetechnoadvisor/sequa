from __future__ import annotations

import datetime
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from sequa.models import Cassette


@dataclass
class SearchResult:
    id: str
    hash: str
    provider: str
    model: str
    created_at: str
    latency_ms: float | None
    file_path: str
    score: float
    input_snippet: str
    output_snippet: str
    cassette: Cassette | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hash": self.hash,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "latency_ms": self.latency_ms,
            "file_path": self.file_path,
            "score": round(self.score, 4),
            "input_snippet": self.input_snippet,
            "output_snippet": self.output_snippet,
        }


def extract_searchable_text(data_or_cassette: dict[str, Any] | Cassette) -> tuple[str, str, str]:
    """Extracts (full_searchable_text, input_snippet, output_snippet) from a Cassette object or dict."""
    if isinstance(data_or_cassette, Cassette):
        data = data_or_cassette.to_dict()
    else:
        data = data_or_cassette

    req = data.get("request", {})
    res = data.get("response", {})

    inputs: list[str] = []
    outputs: list[str] = []

    # Extract inputs
    messages = req.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    inputs.append(content.strip())
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            inputs.append(part.get("text", ""))

    prompt = req.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        inputs.append(prompt.strip())
    elif isinstance(prompt, list):
        for p in prompt:
            if isinstance(p, str):
                inputs.append(p)

    req_input = req.get("input")
    if isinstance(req_input, str) and req_input.strip():
        inputs.append(req_input.strip())

    # Extract outputs
    out = res.get("output")
    if isinstance(out, str) and out.strip():
        outputs.append(out.strip())
    elif isinstance(out, list):
        for o in out:
            if isinstance(o, str):
                outputs.append(o)

    choices = res.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        outputs.append(content.strip())
                text = choice.get("text")
                if isinstance(text, str) and text.strip():
                    outputs.append(text.strip())

    res_content = res.get("content")
    if isinstance(res_content, str) and res_content.strip():
        outputs.append(res_content.strip())

    input_text = " ".join(inputs)
    output_text = " ".join(outputs)
    full_text = f"{input_text} {output_text}".strip()

    input_snippet = input_text[:120] + "..." if len(input_text) > 120 else input_text
    output_snippet = output_text[:120] + "..." if len(output_text) > 120 else output_text

    return full_text, input_snippet, output_snippet


def parse_time_constraint(time_str: str | None) -> datetime.datetime | None:
    """Parses relative time strings ('10m', '2h', '1d', '7d', 'yesterday') or ISO strings into UTC datetime."""
    if not time_str:
        return None

    time_str = time_str.strip()
    now = datetime.datetime.now(datetime.timezone.utc)

    # Relative shortcuts
    lower = time_str.lower()
    if lower == "yesterday":
        return now - datetime.timedelta(days=1)
    if lower == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    m = re.match(r"^(\d+)\s*([smhdw])$", lower)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if unit == "s":
            return now - datetime.timedelta(seconds=val)
        elif unit == "m":
            return now - datetime.timedelta(minutes=val)
        elif unit == "h":
            return now - datetime.timedelta(hours=val)
        elif unit == "d":
            return now - datetime.timedelta(days=val)
        elif unit == "w":
            return now - datetime.timedelta(weeks=val)

    # Try ISO parsing
    try:
        dt = datetime.datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except ValueError:
        pass

    return None


def parse_created_at(created_at_str: str) -> datetime.datetime | None:
    """Parses stored created_at timestamp into a UTC datetime object."""
    if not created_at_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(created_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except ValueError:
        return None


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Calculates cosine similarity between two numeric vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class TFIDFEmbedder:
    """Lightweight, zero-dependency TF-IDF + Cosine Similarity Vectorizer."""

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit_transform(self, corpus: list[str]) -> list[dict[str, float]]:
        tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        doc_count = len(corpus)

        # Calculate Document Frequency (DF)
        df: Counter[str] = Counter()
        for tokens in tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        # Calculate IDF
        idf: dict[str, float] = {}
        for token, count in df.items():
            idf[token] = math.log((1 + doc_count) / (1 + count)) + 1.0

        # Calculate TF-IDF vectors
        vectors: list[dict[str, float]] = []
        for tokens in tokenized_corpus:
            tf = Counter(tokens)
            total_tokens = len(tokens) or 1
            vec: dict[str, float] = {}
            for token, freq in tf.items():
                vec[token] = (freq / total_tokens) * idf[token]
            vectors.append(vec)

        self.idf = idf
        return vectors

    def transform_query(self, query: str) -> dict[str, float]:
        tokens = self.tokenize(query)
        tf = Counter(tokens)
        total_tokens = len(tokens) or 1
        vec: dict[str, float] = {}
        idf = getattr(self, "idf", {})
        for token, freq in tf.items():
            token_idf = idf.get(token, 1.0)
            vec[token] = (freq / total_tokens) * token_idf
        return vec

    def dict_cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[k] * vec2[k] for k in intersection)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


def search_cassettes(
    query: str = "",
    path: str = "cassettes",
    since: str | None = None,
    until: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_k: int = 5,
    storage: Any = None,
) -> list[SearchResult]:
    """Search stored cassettes by time range, provider/model metadata, and cosine similarity query."""
    from sequa.cli.main import load_all_cassettes
    from sequa.storage import FileStorage, MemoryStorage

    # 1. Parse time constraints
    dt_since = parse_time_constraint(since)
    dt_until = parse_time_constraint(until)

    # 2. Gather cassettes
    raw_cassettes: list[tuple[str, dict[str, Any]]] = []
    if storage is not None:
        if isinstance(storage, MemoryStorage):
            for key in storage.list():
                c = storage.load(key)
                raw_cassettes.append((key, c.to_dict()))
        elif isinstance(storage, FileStorage):
            for key in storage.list():
                c = storage.load(key)
                raw_cassettes.append((key, c.to_dict()))
    else:
        raw_cassettes = load_all_cassettes(path)

    if not raw_cassettes:
        return []

    # 3. Filter candidates by time & metadata
    candidates: list[dict[str, Any]] = []

    for file_path, data in raw_cassettes:
        created_at_str = data.get("created_at", "")
        dt_created = parse_created_at(created_at_str)

        if dt_since and dt_created and dt_created < dt_since:
            continue
        if dt_until and dt_created and dt_created > dt_until:
            continue

        c_provider = data.get("provider", "")
        if provider and provider.lower() not in c_provider.lower():
            continue

        req = data.get("request", {})
        c_model = req.get("model") or ""
        if model and model.lower() not in c_model.lower():
            continue

        full_text, in_snip, out_snip = extract_searchable_text(data)

        latency = data.get("metadata", {}).get("latency_ms")
        if latency is None:
            latency = data.get("response", {}).get("latency")

        candidates.append({
            "id": data.get("id", ""),
            "hash": data.get("hash", os.path.basename(file_path).replace(".json", "")),
            "provider": c_provider or "unknown",
            "model": c_model or "unknown",
            "created_at": created_at_str,
            "latency_ms": float(latency) if latency is not None else None,
            "file_path": file_path,
            "full_text": full_text,
            "input_snippet": in_snip,
            "output_snippet": out_snip,
            "raw_data": data,
        })

    if not candidates:
        return []

    # 4. If query is empty, sort by created_at descending
    if not query.strip():
        def sort_key(c: dict[str, Any]) -> str:
            return c["created_at"] or ""

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        results: list[SearchResult] = []
        for item in sorted_candidates[:top_k]:
            results.append(SearchResult(
                id=item["id"],
                hash=item["hash"],
                provider=item["provider"],
                model=item["model"],
                created_at=item["created_at"],
                latency_ms=item["latency_ms"],
                file_path=item["file_path"],
                score=1.0,
                input_snippet=item["input_snippet"],
                output_snippet=item["output_snippet"],
                cassette=Cassette.from_dict(item["raw_data"]),
            ))
        return results

    # 5. Perform TF-IDF Cosine Similarity Ranking
    corpus = [item["full_text"] for item in candidates]
    embedder = TFIDFEmbedder()
    doc_vectors = embedder.fit_transform(corpus)
    query_vector = embedder.transform_query(query)

    scored_items: list[tuple[float, dict[str, Any]]] = []
    for idx, item in enumerate(candidates):
        score = embedder.dict_cosine_similarity(query_vector, doc_vectors[idx])
        scored_items.append((score, item))

    # Sort descending by score
    scored_items.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, item in scored_items:
        if score <= 0.0:
            break
        results.append(SearchResult(
            id=item["id"],
            hash=item["hash"],
            provider=item["provider"],
            model=item["model"],
            created_at=item["created_at"],
            latency_ms=item["latency_ms"],
            file_path=item["file_path"],
            score=score,
            input_snippet=item["input_snippet"],
            output_snippet=item["output_snippet"],
            cassette=Cassette.from_dict(item["raw_data"]),
        ))
        if len(results) >= top_k:
            break


    return results
