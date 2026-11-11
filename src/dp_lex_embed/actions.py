"""Proc-protocol v1.1 actions of dp-lex-embed: spec, check, run.

Dependencies (run opener, embedder factory, schema inspector) are injected; `main.py` wires Postgres and Jina. Failures are
`ActionError`s carrying one of the three protocol error codes; the connection string and the Jina key never appear in a message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import v11
from dp_lex_embed.identifiers import InvalidIdentifierError, check_ident, parse_table
from dp_lex_embed.model import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, EmbedOptions, EmbedResult
from dp_lex_embed.service import EmbedService

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "connString": {"type": "string", "description": "Dataset connection string (${dataset.connString}, rw role)."},
        "table": {"type": "string", "description": "schema.table to embed."},
        "keyColumn": {"type": "string"},
        "textColumns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "vectorColumn": {"type": "string"},
        "model": {"type": "string", "default": "jina-embeddings-v3"},
        "tokenBudget": {"type": "integer", "minimum": 0},
        "batchSize": {"type": "integer", "minimum": 1, "maximum": 64, "default": 16},
        "jinaApiKey": {"type": "string", "description": "Jina API key (${secret.JINA_API_KEY})."},
        "items": {"type": "array", "items": {"type": "object"}},
        "sourceId": {"type": "string"},
        "batchId": {"type": "string"},
        "packageVersion": {"type": "string"},
        "dryRun": {"type": "boolean", "default": False},
    },
    "required": ["connString", "table", "keyColumn", "textColumns", "vectorColumn", "tokenBudget", "jinaApiKey"],
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "metrics": {"type": "object", "additionalProperties": {"type": "number"}},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "severity": {"enum": ["info", "warn", "error"]},
                    "message": {"type": "string"},
                },
                "required": ["name", "passed", "severity"],
            },
        },
    },
    "required": ["metrics", "checks"],
}


class ActionError(Exception):
    """A protocol error: one of the three codes, a message without secrets, `retry_after_sec` for transient errors."""

    def __init__(self, code: v11.ProcErrorCode, message: str, retry_after_sec: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_sec = retry_after_sec


def config_error(message: str) -> ActionError:
    return ActionError(v11.ProcErrorCode.CONFIG_ERROR, message)


MAX_BATCH_SIZE = 64


def metrics_for(table: str, *, pending: int = 0, embedded: int = 0, skipped: int = 0, tokens: int = 0) -> dict[str, float]:
    return {"pending": pending, "embedded": embedded, "skipped": skipped, "embed_tokens": tokens, f"written.{table}": embedded}


class Actions:
    def __init__(self, open_run: Callable[[str, EmbedOptions], Any], embedder_factory: Callable[[str], Any], inspector: Any) -> None:
        self._open_run = open_run
        self._embedder_factory = embedder_factory
        self._inspector = inspector

    @staticmethod
    def spec() -> dict[str, Any]:
        return v11.spec_result(CONFIG_SCHEMA, OUTPUT_SCHEMA)

    def check(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if params.get("dryRun") is True:  # conformance at registration: no database, no Jina
            return {"ok": True}
        return {"ok": True}

    def run(self, params: Mapping[str, Any], progress: Callable[[int, str], None] | None = None) -> dict[str, Any]:
        dry_run = params.get("dryRun") is True
        if dry_run and not params.get("connString"):  # conformance at registration: nothing to read, nothing to call
            schema, name = _table_of(params)
            return v11.run_result(metrics_for(f"{schema}.{name}"), [])
        options = parse_options(params)
        conn_string = _required_text(params, "connString")
        if dry_run:
            with self._open_run(conn_string, options) as repo:
                plan = EmbedService(repo, _NoEmbedder(), options).plan()
            return v11.run_result(metrics_for(options.table, pending=plan.pending, tokens=plan.estimated_tokens), [])
        embedder = self._embedder_factory(_required_text(params, "jinaApiKey"))
        with self._open_run(conn_string, options) as repo:
            outcome = EmbedService(repo, embedder, options).run(_batch_progress(progress))
        metrics = metrics_for(
            options.table, pending=outcome.pending, embedded=outcome.embedded, skipped=outcome.skipped, tokens=outcome.tokens,
        )
        checks = [{
            "name": "budget_not_exhausted", "passed": outcome.left == 0, "severity": "warn",
            "message": f"{outcome.left} row(s) still without a vector (tokenBudget {options.token_budget})",
        }]
        return v11.run_result(metrics, checks)


class _NoEmbedder:
    @staticmethod
    def embed_passages(texts: list[str]) -> EmbedResult:  # noqa: ARG004
        msg = "dryRun never embeds"
        raise AssertionError(msg)


def _required_text(params: Mapping[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise config_error(f"{name} is required")
    return value


def _batch_progress(progress: Callable[[int, str], None] | None) -> Callable[[int, int], None] | None:
    if progress is None:
        return None

    def on_batch(done: int, total: int) -> None:
        progress(int(100 * done / total) if total else 100, f"embedded {done}/{total} rows")

    return on_batch


def _int(params: Mapping[str, Any], name: str, default: int | None = None) -> int:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise config_error(f"{name} must be a non-negative integer")
    return value


def parse_options(params: Mapping[str, Any]) -> EmbedOptions:
    """Validates the config into options; every identifier is checked (P78b-01)."""
    columns = params.get("textColumns")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise config_error("textColumns must be a non-empty list of column names")
    batch_size = _int(params, "batchSize", DEFAULT_BATCH_SIZE)
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise config_error(f"batchSize must be between 1 and {MAX_BATCH_SIZE}")
    model = params.get("model", DEFAULT_MODEL)
    if model != DEFAULT_MODEL:
        raise config_error(f"model {model!r} is not supported (only {DEFAULT_MODEL})")
    try:
        table = params.get("table")
        schema, name = parse_table(table if isinstance(table, str) else "")
        key = check_ident(str(params.get("keyColumn")))
        vector = check_ident(str(params.get("vectorColumn")))
        texts = tuple(check_ident(c) for c in columns)
    except InvalidIdentifierError as ex:
        raise config_error(str(ex)) from None
    return EmbedOptions(f"{schema}.{name}", key, texts, vector, _int(params, "tokenBudget"), batch_size, DEFAULT_MODEL)


def _table_of(params: Mapping[str, Any]) -> tuple[str, str]:
    table = params.get("table")
    try:
        return parse_table(table if isinstance(table, str) else "")
    except InvalidIdentifierError as ex:
        raise config_error(str(ex)) from None
