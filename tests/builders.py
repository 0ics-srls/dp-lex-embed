"""In-memory fakes for the embed service tests."""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from dp_lex_embed.model import EmbedOptions, EmbedResult, Row, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Iterator


class FakeRepository:
    def __init__(self, texts: dict[int, str], vectors: dict[int, list[float]] | None = None) -> None:
        self.texts = dict(texts)
        self.vectors: dict[int, list[float]] = dict(vectors or {})
        self.store_calls = 0

    def count_pending(self) -> int:
        return sum(1 for k in self.texts if k not in self.vectors)

    def fetch_pending(self, after_key: object, limit: int) -> list[Row]:
        keys = sorted(k for k in self.texts if k not in self.vectors and (after_key is None or k > after_key))  # type: ignore[operator]
        return [Row(k, self.texts[k]) for k in keys[:limit]]

    def store(self, items: list[tuple[object, list[float]]]) -> int:
        self.store_calls += 1
        written = 0
        for key, vector in items:
            if key not in self.vectors:
                self.vectors[key] = vector  # type: ignore[index]
                written += 1
        return written


class FakeEmbedder:
    """Tokens = the service estimate; optionally fails on the n-th call (1-based)."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            msg = "boom"
            raise RuntimeError(msg)
        return EmbedResult([[float(len(t))] for t in texts], sum(estimate_tokens(t) for t in texts))


def options(**overrides: Any) -> EmbedOptions:
    values: dict[str, Any] = {
        "table": "ds_x.notes", "key_column": "id", "text_columns": ("body",), "vector_column": "embedding",
        "token_budget": 10_000, "batch_size": 2,
    }
    values.update(overrides)
    return EmbedOptions(**values)


class FakeRunOpener:
    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository

    @contextmanager
    def __call__(self, _conn: str, _options: EmbedOptions) -> Iterator[FakeRepository]:
        yield self.repository
