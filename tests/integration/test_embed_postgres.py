"""Real Postgres + pgvector (Testcontainers); the embedder is a fake (no Jina call)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from testcontainers.community.postgres import PostgresContainer

from dp_lex_embed.actions import Actions
from dp_lex_embed.embeddings import DIMENSIONS
from dp_lex_embed.model import EmbedResult, estimate_tokens
from dp_lex_embed.pg_repository import PgInspector, open_run

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration

PASSWORD = "pw"


class _Embedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        self.texts.extend(texts)
        return EmbedResult([[0.5] * DIMENSIONS for _ in texts], sum(estimate_tokens(t) for t in texts))


class _Db:
    def __init__(self, container: PostgresContainer) -> None:
        self.dsn = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        self.conn_string = (
            f"Host={container.get_container_host_ip()};Port={container.get_exposed_port(5432)};"
            f"Database={container.dbname};Username={container.username};Password={container.password}"
        )

    def scalar(self, sql: str) -> Any:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            return conn.execute(sql).fetchone()[0]  # type: ignore[index]

    def execute(self, sql: str) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(sql)


@pytest.fixture(scope="module")
def container() -> Iterator[PostgresContainer]:
    with PostgresContainer("pgvector/pgvector:pg16", username="lex", password=PASSWORD, dbname="lex") as pg:
        yield pg


@pytest.fixture
def db(container: PostgresContainer) -> _Db:
    database = _Db(container)
    database.execute("CREATE EXTENSION IF NOT EXISTS vector")
    database.execute("DROP SCHEMA IF EXISTS ds_t CASCADE")
    database.execute("CREATE SCHEMA ds_t")
    database.execute("CREATE TABLE ds_t.notes(id int PRIMARY KEY, title text, body text, embedding vector(1024))")
    rows = ", ".join(f"({i}, 'T{i}', 'body number {i}', NULL)" for i in range(1, 6))
    database.execute(f"INSERT INTO ds_t.notes VALUES {rows}")  # noqa: S608 - test data
    database.execute("INSERT INTO ds_t.notes VALUES (6, NULL, '   ', NULL)")
    database.execute(f"INSERT INTO ds_t.notes VALUES (7, 'T7', 'already', '[{','.join(['0.1'] * DIMENSIONS)}]')")  # noqa: S608
    return database


def _params(db: _Db, **extra: Any) -> dict[str, Any]:
    return {"connString": db.conn_string, "jinaApiKey": "k", "table": "ds_t.notes", "keyColumn": "id", "textColumns": ["body"],
            "vectorColumn": "embedding", "tokenBudget": 10_000, "batchSize": 2, **extra}


def _actions(embedder: _Embedder) -> Actions:
    return Actions(open_run, lambda _: embedder, PgInspector())


def test_should_count_pending_rows_and_estimate_tokens_when_dry_run(db: _Db) -> None:
    embedder = _Embedder()

    metrics = _actions(embedder).run(_params(db, dryRun=True), None)["metrics"]

    assert metrics["pending"] == 6
    assert metrics["embed_tokens"] > 0
    assert metrics["embedded"] == 0
    assert embedder.texts == []
    assert db.scalar("SELECT count(*) FROM ds_t.notes WHERE embedding IS NULL") == 6


def test_should_fill_only_null_vectors_and_write_nothing_on_the_second_run(db: _Db) -> None:
    embedder = _Embedder()

    first = _actions(embedder).run(_params(db), None)["metrics"]
    second = _actions(embedder).run(_params(db), None)["metrics"]

    assert first["embedded"] == 5
    assert first["skipped"] == 1
    assert first["written.ds_t.notes"] == 5
    assert second["embedded"] == 0
    assert db.scalar("SELECT count(*) FROM ds_t.notes WHERE embedding IS NOT NULL") == 6
    assert db.scalar("SELECT (embedding::real[])[1] FROM ds_t.notes WHERE id = 7") == pytest.approx(0.1)


def test_should_stop_early_when_the_token_budget_is_small(db: _Db) -> None:
    embedder = _Embedder()

    metrics = _actions(embedder).run(_params(db, tokenBudget=10), None)["metrics"]

    assert 0 < metrics["embedded"] < 5
    assert metrics["embed_tokens"] <= 10


def test_should_join_several_text_columns_with_a_newline(db: _Db) -> None:
    embedder = _Embedder()

    _actions(embedder).run(_params(db, textColumns=["title", "body"]), None)

    assert "T1\nbody number 1" in embedder.texts


def test_should_pass_check_on_the_real_table_and_fail_on_a_wrong_vector_column(db: _Db) -> None:
    assert _actions(_Embedder()).check(_params(db)) == {"ok": True}
    with pytest.raises(Exception, match="expected vector"):
        _actions(_Embedder()).check(_params(db, vectorColumn="title"))
