"""Passage embeddings with Jina v3 (adapted from dp-letaky-match; batch of texts, token usage from the response).

Transient failures (429, 5xx, network) are retried with the server's Retry-After, else exponential backoff with full jitter capped at
30 s; other 4xx are fatal (P78b-04). The API key never appears in an exception message or a log line.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import httpx

from dp_lex_embed.model import EmbedResult, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable
    from random import Random

log = logging.getLogger(__name__)

ENDPOINT = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v3"
TASK = "retrieval.passage"
DIMENSIONS = 1024
MAX_CHARS_PER_INPUT = 8000
MAX_ATTEMPTS = 6
BACKOFF_BASE_SEC = 0.25
BACKOFF_MAX_SEC = 30.0
DETAIL_LIMIT = 500
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500


class EmbeddingUnavailable(Exception):  # noqa: N818 - domain name; the transient counterpart of EmbeddingRejected
    """Jina could not be reached or kept answering 429 / 5xx; the step ends with a transient error."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingRejected(Exception):  # noqa: N818 - domain name
    """Jina answered with a client error or a malformed vector: retrying will not help."""


def _split_to_cap(text: str, cap: int = MAX_CHARS_PER_INPUT) -> list[str]:
    """Pieces of at most `cap` characters, split after the last whitespace inside the window."""
    if len(text) <= cap:
        return [text]
    pieces: list[str] = []
    position = 0
    while position < len(text):
        length = min(cap, len(text) - position)
        if position + length < len(text):
            last_space = text.rfind(" ", position, position + length)
            if last_space > position:
                length = last_space - position + 1
        piece = text[position:position + length]
        position += length
        if piece.strip():
            pieces.append(piece)
    return pieces


def _mean_pool_normalised(vectors: list[list[float]]) -> list[float]:
    pooled = [sum(column) / len(vectors) for column in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(x * x for x in pooled))
    return [x / norm for x in pooled] if norm > 0 else pooled


def _retry_after(response: httpx.Response) -> int | None:
    value = response.headers.get("Retry-After", "").strip()
    return int(float(value)) if value.replace(".", "", 1).isdigit() else None


class JinaEmbeddingGenerator:
    def __init__(self, api_key: str, client: httpx.Client, sleep: Callable[[float], None], rng: Random) -> None:
        self._api_key = api_key
        self._client = client
        self._sleep = sleep
        self._rng = rng

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        pieces: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            parts = _split_to_cap(text)
            spans.append((len(pieces), len(pieces) + len(parts)))
            pieces.extend(parts)
        vectors, reported = self._embed(pieces)
        pooled = [vectors[a] if b - a == 1 else _mean_pool_normalised(vectors[a:b]) for a, b in spans]
        tokens = reported if reported is not None else sum(estimate_tokens(p) for p in pieces)
        return EmbedResult(pooled, tokens)

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _backoff(self, attempt: int) -> float:
        return float(self._rng.random() * min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * 2**attempt))

    def _embed(self, pieces: list[str]) -> tuple[list[list[float]], int | None]:
        body = {"model": MODEL, "task": TASK, "dimensions": DIMENSIONS, "input": pieces}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_status: str = "no answer"
        retry_after: int | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.post(ENDPOINT, json=body, headers=headers)
            except httpx.TransportError as ex:
                last_status, retry_after = type(ex).__name__, None
            else:
                if response.status_code == httpx.codes.OK:
                    return self._parse(response, len(pieces))
                if response.status_code != HTTP_TOO_MANY_REQUESTS and response.status_code < HTTP_SERVER_ERROR:
                    detail = self._scrub(response.text)[:DETAIL_LIMIT]
                    msg = f"Jina rejected the request (HTTP {response.status_code}): {detail}"
                    raise EmbeddingRejected(msg)
                last_status, retry_after = f"HTTP {response.status_code}", _retry_after(response)
            if attempt + 1 < MAX_ATTEMPTS:
                delay = float(retry_after) if retry_after is not None else self._backoff(attempt)
                log.warning("Jina attempt %s/%s failed (%s); retrying in %.1fs", attempt + 1, MAX_ATTEMPTS, last_status, delay)
                self._sleep(delay)
        msg = f"Jina unavailable after {MAX_ATTEMPTS} attempts (last: {last_status})"
        raise EmbeddingUnavailable(msg, retry_after)

    @staticmethod
    def _parse(response: httpx.Response, expected: int) -> tuple[list[list[float]], int | None]:
        try:
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in data]
            usage = payload.get("usage")
            tokens = int(usage["total_tokens"]) if isinstance(usage, dict) and "total_tokens" in usage else None
        except (KeyError, TypeError, ValueError) as ex:
            msg = f"Jina returned a malformed response ({type(ex).__name__})"
            raise EmbeddingRejected(msg) from None
        if len(vectors) != expected or any(len(v) != DIMENSIONS for v in vectors):
            msg = f"Jina returned {len(vectors)} vector(s), expected {expected} of {DIMENSIONS} dimensions"
            raise EmbeddingRejected(msg)
        return vectors, tokens
