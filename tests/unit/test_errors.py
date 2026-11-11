"""Error mapping of the actions (RUN-9) and the check action."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import v11
from builders import FakeRepository, FakeRunOpener
from dp_lex_embed.actions import ActionError, Actions
from dp_lex_embed.embeddings import EmbeddingRejected, EmbeddingUnavailable
from dp_lex_embed.errors import DbConnectError

pytestmark = pytest.mark.unit

CONN = "Host=h;Username=u;Password=SECRET-PW"
KEY = "jina_SECRET-KEY-123"
FULL = {"connString": CONN, "jinaApiKey": KEY, "table": "ds_x.notes", "keyColumn": "id", "textColumns": ["body"],
        "vectorColumn": "embedding", "tokenBudget": 100}


class _FailingEmbedder:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def embed_passages(self, texts: list[str]) -> Any:  # noqa: ARG002
        raise self.error


def _failing(error: Exception) -> Actions:
    return Actions(FakeRunOpener(FakeRepository({1: "aaaaaa"})), lambda _: _FailingEmbedder(error), None)


def test_should_map_unavailable_to_transient_with_retry_after() -> None:
    with pytest.raises(ActionError) as ex:
        _failing(EmbeddingUnavailable(f"down {KEY}", 42)).run(FULL, None)

    assert ex.value.code is v11.ProcErrorCode.TRANSIENT_ERROR
    assert ex.value.retry_after_sec == 42
    assert KEY not in str(ex.value)


def test_should_map_rejected_to_config_error() -> None:
    with pytest.raises(ActionError) as ex:
        _failing(EmbeddingRejected(f"HTTP 401 {KEY}")).run(FULL, None)

    assert ex.value.code is v11.ProcErrorCode.CONFIG_ERROR
    assert KEY not in str(ex.value)


def test_should_map_an_unexpected_failure_to_system_error_without_secrets() -> None:
    with pytest.raises(ActionError) as ex:
        _failing(RuntimeError(f"boom {CONN} {KEY}")).run(FULL, None)

    assert ex.value.code is v11.ProcErrorCode.SYSTEM_ERROR
    assert CONN not in str(ex.value)
    assert KEY not in str(ex.value)


def _opener_failing(error: DbConnectError) -> Any:
    @contextmanager
    def opener(_conn: str, _options: Any) -> Any:
        raise error
        yield  # pragma: no cover

    return opener


def test_should_map_a_rejected_login_to_config_error_and_an_unreachable_database_to_transient() -> None:
    rejected = Actions(_opener_failing(DbConnectError("login", sqlstate="28P01")), lambda _: None, None)
    unreachable = Actions(_opener_failing(DbConnectError("down")), lambda _: None, None)

    with pytest.raises(ActionError) as login:
        rejected.run(FULL, None)
    with pytest.raises(ActionError) as down:
        unreachable.run(FULL, None)

    assert login.value.code is v11.ProcErrorCode.CONFIG_ERROR
    assert down.value.code is v11.ProcErrorCode.TRANSIENT_ERROR


class _Inspector:
    def __init__(self, columns: dict[str, str]) -> None:
        self._columns = columns

    def columns(self, conn_string: str, schema: str, table: str) -> dict[str, str]:  # noqa: ARG002
        return self._columns


def _check(columns: dict[str, str]) -> Any:
    def no_embedder(_: str) -> Any:
        msg = "check must not call Jina"
        raise AssertionError(msg)

    return Actions(lambda *_: None, no_embedder, _Inspector(columns)).check(FULL)


def test_should_pass_check_when_columns_and_a_1024_vector_exist_without_calling_jina() -> None:
    assert _check({"id": "integer", "body": "text", "embedding": "vector(1024)"}) == {"ok": True}


@pytest.mark.parametrize("columns", [
    {},
    {"id": "integer", "body": "text"},
    {"id": "integer", "body": "text", "embedding": "vector(768)"},
    {"id": "integer", "body": "text", "embedding": "text"},
])
def test_should_fail_check_with_config_error_when_table_or_columns_are_wrong(columns: dict[str, str]) -> None:
    with pytest.raises(ActionError) as ex:
        _check(columns)

    assert ex.value.code is v11.ProcErrorCode.CONFIG_ERROR
