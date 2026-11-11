from __future__ import annotations

import pytest

from builders import FakeEmbedder, FakeRepository, options
from dp_lex_embed.service import EmbedService

pytestmark = pytest.mark.unit

TEXTS = {1: "aaaaaa", 2: "bbbbbb", 3: "cccccc", 4: "dddddd", 5: "eeeeee"}  # 6 chars = 2 estimated tokens each


def test_should_embed_only_rows_without_a_vector() -> None:
    repo = FakeRepository(TEXTS, {1: [9.0], 2: [9.0]})
    embedder = FakeEmbedder()

    outcome = EmbedService(repo, embedder, options()).run()

    assert outcome.embedded == 3
    assert repo.vectors[1] == [9.0]
    assert sum(len(c) for c in embedder.calls) == 3


def test_should_embed_nothing_when_run_a_second_time() -> None:
    repo = FakeRepository(TEXTS)
    EmbedService(repo, FakeEmbedder(), options()).run()
    embedder = FakeEmbedder()

    outcome = EmbedService(repo, embedder, options()).run()

    assert outcome.embedded == 0
    assert outcome.pending == 0
    assert embedder.calls == []


def test_should_stop_before_overshooting_the_token_budget() -> None:
    repo = FakeRepository(TEXTS)

    outcome = EmbedService(repo, FakeEmbedder(), options(token_budget=5)).run()

    assert outcome.tokens <= 5
    assert outcome.embedded == 2
    assert outcome.left == 3


def test_should_embed_nothing_when_the_budget_is_zero() -> None:
    embedder = FakeEmbedder()

    outcome = EmbedService(FakeRepository(TEXTS), embedder, options(token_budget=0)).run()

    assert outcome.embedded == 0
    assert embedder.calls == []
    assert outcome.left == 5


def test_should_skip_blank_text_without_looping() -> None:
    repo = FakeRepository({1: "aaaaaa", 2: "   ", 3: "cccccc"})

    outcome = EmbedService(repo, FakeEmbedder(), options()).run()

    assert (outcome.embedded, outcome.skipped, outcome.left) == (2, 1, 0)
    assert 2 not in repo.vectors


def test_should_respect_the_batch_size() -> None:
    embedder = FakeEmbedder()

    EmbedService(FakeRepository(TEXTS), embedder, options(batch_size=2)).run()

    assert [len(c) for c in embedder.calls] == [2, 2, 1]


def test_should_keep_the_first_batch_when_the_second_fails() -> None:
    repo = FakeRepository(TEXTS)

    with pytest.raises(RuntimeError):
        EmbedService(repo, FakeEmbedder(fail_on_call=2), options()).run()

    assert sorted(repo.vectors) == [1, 2]


def test_should_only_plan_when_dry_run() -> None:
    repo = FakeRepository(TEXTS, {1: [9.0]})
    embedder = FakeEmbedder()

    plan = EmbedService(repo, embedder, options()).plan()

    assert (plan.pending, plan.estimated_tokens) == (4, 8)
    assert embedder.calls == []
    assert repo.store_calls == 0
