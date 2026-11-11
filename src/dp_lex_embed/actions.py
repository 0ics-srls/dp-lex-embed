"""Proc-protocol v1.1 actions of dp-lex-embed: spec, check, run.

Dependencies (run opener, embedder factory, schema inspector) are injected; `main.py` wires Postgres and Jina. Failures are
`ActionError`s carrying one of the three protocol error codes; the connection string and the Jina key never appear in a message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import v11
from dp_lex_embed.identifiers import InvalidIdentifierError, parse_table

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


def metrics_for(table: str, *, pending: int = 0, embedded: int = 0, skipped: int = 0, tokens: int = 0) -> dict[str, float]:
    return {"pending": pending, "embedded": embedded, "skipped": skipped, "embed_tokens": tokens, f"written.{table}": embedded}


class Actions:
    def __init__(self, open_run: Callable[[str], Any], embedder_factory: Callable[[str], Any], inspector: Any) -> None:
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

    def run(self, params: Mapping[str, Any], progress: Callable[[int, str], None] | None = None) -> dict[str, Any]:  # noqa: ARG002
        table = params.get("table")
        try:
            parse_table(table if isinstance(table, str) else "")
        except InvalidIdentifierError as ex:
            raise config_error(str(ex)) from None
        if params.get("dryRun") is True and not params.get("connString"):
            return v11.run_result(metrics_for(str(table)), [])
        raise config_error("not implemented")
