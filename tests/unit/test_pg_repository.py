from __future__ import annotations

from typing import Any

import pytest

from builders import options
from dp_lex_embed.identifiers import InvalidIdentifierError
from dp_lex_embed.pg_repository import PgEmbedRepository

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    rowcount = 1

    def fetchone(self) -> tuple[Any, ...]:
        return self._rows[0]


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def commit(self) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Cursor:
        self.calls.append((sql, params))
        return _Cursor([(0, "t")])


def test_should_build_quoted_sql_with_bound_values_for_fetch_and_store() -> None:
    conn = _Connection()
    repo = PgEmbedRepository(conn, options(text_columns=("title", "body")))

    repo.count_pending()
    repo.fetch_pending(5, 3)
    repo.fetch_pending(None, 3)
    repo.store([(1, [0.5, 0.25])])

    count, fetch_after, fetch_first, store = (c[0] for c in conn.calls[:4])
    assert 'FROM "ds_x"."notes"' in count
    assert '"embedding" IS NULL' in count
    assert "concat_ws(E'\\n', \"title\", \"body\")" in fetch_after
    assert '"id" > %s' in fetch_after
    assert '"id" >' not in fetch_first
    assert 'ORDER BY "id"' in fetch_after
    assert 'SET "embedding" = %s::vector' in store
    assert 'WHERE "id" = %s AND "embedding" IS NULL' in store
    assert conn.calls[3][1] == ("[0.5,0.25]", 1)


def test_should_reject_a_hostile_identifier_before_any_sql() -> None:
    conn = _Connection()

    with pytest.raises(InvalidIdentifierError):
        PgEmbedRepository(conn, options(vector_column='embedding"; DROP TABLE x; --'))

    assert conn.calls == []
