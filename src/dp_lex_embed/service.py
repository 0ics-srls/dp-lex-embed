"""The embed loop: pending rows in key order, batches cut to the token budget, one commit per batch (P78b-03, P78b-05)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dp_lex_embed.model import EmbedOptions, EmbedResult, Row, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable


class EmbedRepository(Protocol):
    def count_pending(self) -> int: ...

    def fetch_pending(self, after_key: object, limit: int) -> list[Row]:
        """Rows whose vector is NULL with key greater than `after_key` (None = from the start), ordered by key."""
        ...

    def store(self, items: list[tuple[object, list[float]]]) -> int:
        """Writes the vectors where the vector is still NULL and commits; returns the rows written."""
        ...


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> EmbedResult: ...


@dataclass(frozen=True, slots=True)
class Plan:
    pending: int
    estimated_tokens: int


@dataclass(frozen=True, slots=True)
class EmbedOutcome:
    pending: int
    embedded: int
    skipped: int
    tokens: int

    @property
    def left(self) -> int:
        return self.pending - self.embedded - self.skipped


class EmbedService:
    def __init__(self, repository: EmbedRepository, embedder: Embedder, options: EmbedOptions) -> None:
        self._repo = repository
        self._embedder = embedder
        self._options = options

    def plan(self) -> Plan:
        pending = self._repo.count_pending()
        tokens = 0
        after: object = None
        while True:
            rows = self._repo.fetch_pending(after, self._options.batch_size)
            if not rows:
                break
            tokens += sum(estimate_tokens(r.text) for r in rows if r.text.strip())
            after = rows[-1].key
        return Plan(pending, tokens)

    def run(self, progress: Callable[[int, int], None] | None = None) -> EmbedOutcome:
        pending = self._repo.count_pending()
        budget = self._options.token_budget
        embedded = skipped = tokens = 0
        after: object = None
        exhausted = False
        while not exhausted:
            rows = self._repo.fetch_pending(after, self._options.batch_size)
            if not rows:
                break
            batch: list[Row] = []
            estimate = 0
            consumed = 0
            for row in rows:
                if not row.text.strip():
                    skipped += 1
                    consumed += 1
                    continue
                cost = estimate_tokens(row.text)
                if tokens + estimate + cost > budget:
                    exhausted = True
                    break
                batch.append(row)
                estimate += cost
                consumed += 1
            if batch:
                result = self._embedder.embed_passages([r.text for r in batch])
                embedded += self._repo.store([(r.key, v) for r, v in zip(batch, result.vectors, strict=True)])
                tokens += result.tokens
                exhausted = exhausted or tokens >= budget
            if consumed:
                after = rows[consumed - 1].key
            if progress is not None:
                progress(embedded + skipped, pending)
        return EmbedOutcome(pending, embedded, skipped, tokens)
