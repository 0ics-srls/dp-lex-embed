from __future__ import annotations

import json
import random
from typing import Any

import httpx
import pytest

from dp_lex_embed.embeddings import DIMENSIONS, EmbeddingRejected, EmbeddingUnavailable, JinaEmbeddingGenerator

pytestmark = pytest.mark.unit

KEY = "jina_SECRET-KEY-123"


def _vec(value: float) -> list[float]:
    return [value] * DIMENSIONS


def _ok(count: int, tokens: int | None = 7, *, reverse: bool = False) -> httpx.Response:
    data = [{"index": i, "embedding": _vec(float(i + 1))} for i in range(count)]
    if reverse:
        data.reverse()
    body: dict[str, Any] = {"data": data}
    if tokens is not None:
        body["usage"] = {"total_tokens": tokens}
    return httpx.Response(200, json=body)


class _Server:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _generator(server: _Server, sleeps: list[float] | None = None) -> JinaEmbeddingGenerator:
    client = httpx.Client(transport=httpx.MockTransport(server))
    return JinaEmbeddingGenerator(KEY, client, (sleeps if sleeps is not None else []).append, random.Random(1))  # noqa: S311


def test_should_return_vectors_in_index_order_and_the_reported_tokens() -> None:
    result = _generator(_Server(_ok(2, tokens=11, reverse=True))).embed_passages(["a", "b"])

    assert [v[0] for v in result.vectors] == [1.0, 2.0]
    assert result.tokens == 11


def test_should_estimate_tokens_when_usage_is_missing() -> None:
    result = _generator(_Server(_ok(1, tokens=None))).embed_passages(["abcdef"])

    assert result.tokens == 2


def test_should_send_model_task_dimensions_and_the_bearer_key() -> None:
    server = _Server(_ok(1))

    _generator(server).embed_passages(["hello"])

    request = server.requests[0]
    body = json.loads(request.content)
    assert body["model"] == "jina-embeddings-v3"
    assert body["task"] == "retrieval.passage"
    assert body["dimensions"] == 1024
    assert body["input"] == ["hello"]
    assert request.headers["Authorization"] == f"Bearer {KEY}"


def test_should_retry_once_after_a_429_with_retry_after() -> None:
    sleeps: list[float] = []
    server = _Server(httpx.Response(429, headers={"Retry-After": "3"}), _ok(1))

    _generator(server, sleeps).embed_passages(["a"])

    assert len(server.requests) == 2
    assert sleeps == [3.0]


def test_should_raise_unavailable_after_six_attempts_on_500() -> None:
    server = _Server(httpx.Response(500, text="oops"))
    sleeps: list[float] = []

    with pytest.raises(EmbeddingUnavailable):
        _generator(server, sleeps).embed_passages(["a"])

    assert len(server.requests) == 6
    assert len(sleeps) == 5
    assert all(0 <= s <= 30 for s in sleeps)


def test_should_raise_rejected_without_retry_on_401_and_not_leak_the_key() -> None:
    server = _Server(httpx.Response(401, text=f"bad key {KEY}"))

    with pytest.raises(EmbeddingRejected) as ex:
        _generator(server).embed_passages(["a"])

    assert len(server.requests) == 1
    assert KEY not in str(ex.value)


def test_should_split_a_long_text_and_mean_pool_its_pieces() -> None:
    server = _Server(_ok(2, tokens=20))
    long_text = ("word " * 2000).strip()  # 9999 chars -> 2 pieces

    result = _generator(server).embed_passages([long_text])

    assert len(json.loads(server.requests[0].content)["input"]) == 2
    assert len(result.vectors) == 1
    assert result.tokens == 20
    assert sum(x * x for x in result.vectors[0]) == pytest.approx(1.0)


def test_should_reject_a_vector_with_the_wrong_dimension() -> None:
    bad = httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    with pytest.raises(EmbeddingRejected):
        _generator(_Server(bad)).embed_passages(["a"])
