"""Postgres side of the embed step: pending rows by keyset paging, vectors written as pgvector text literals (P78b-01, P78b-02)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import psycopg

from dp_lex_embed.db import connect
from dp_lex_embed.errors import RepositoryError
from dp_lex_embed.identifiers import parse_table, quote_ident
from dp_lex_embed.model import Row

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dp_lex_embed.model import EmbedOptions


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgEmbedRepository:
    """All identifiers are validated and quoted when the repository is built, values are always bound parameters."""

    def __init__(self, connection: Any, options: EmbedOptions) -> None:
        schema, table = parse_table(options.table)
        self._conn = connection
        self._table = f"{quote_ident(schema)}.{quote_ident(table)}"
        self._key = quote_ident(options.key_column)
        self._vector = quote_ident(options.vector_column)
        self._text = "concat_ws(E'\\n', " + ", ".join(quote_ident(c) for c in options.text_columns) + ")"

    def count_pending(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self._table} WHERE {self._vector} IS NULL").fetchone()  # noqa: S608 - quoted identifiers
        return int(row[0])

    def fetch_pending(self, after_key: object, limit: int) -> list[Row]:
        after = "" if after_key is None else f" AND {self._key} > %s"
        sql = (
            f"SELECT {self._key}, {self._text} FROM {self._table} "  # noqa: S608 - quoted identifiers, values bound
            f"WHERE {self._vector} IS NULL{after} ORDER BY {self._key} LIMIT %s"
        )
        params: tuple[Any, ...] = (limit,) if after_key is None else (after_key, limit)
        return [Row(r[0], r[1] or "") for r in self._conn.execute(sql, params).fetchall()]

    def store(self, items: list[tuple[object, list[float]]]) -> int:
        sql = f"UPDATE {self._table} SET {self._vector} = %s::vector WHERE {self._key} = %s AND {self._vector} IS NULL"  # noqa: S608
        written = 0
        try:
            for key, vector in items:
                written += self._conn.execute(sql, (_literal(vector), key)).rowcount
            self._conn.commit()
        except psycopg.Error as ex:
            self._conn.rollback()
            msg = f"writing the vectors failed ({type(ex).__name__})"
            raise RepositoryError(msg) from None
        return written


@contextmanager
def open_run(conn_string: str, options: EmbedOptions) -> Iterator[PgEmbedRepository]:
    connection = connect(conn_string)
    try:
        yield PgEmbedRepository(connection, options)
    finally:
        connection.close()


class PgInspector:
    """Column name -> formatted type of a table (empty when the table does not exist), for the `check` action."""

    @staticmethod
    def columns(conn_string: str, schema: str, table: str) -> dict[str, str]:
        connection = connect(conn_string)
        try:
            rows = connection.execute(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped", (schema, table),
            ).fetchall()
            return {str(r[0]): str(r[1]) for r in rows}
        finally:
            connection.close()
