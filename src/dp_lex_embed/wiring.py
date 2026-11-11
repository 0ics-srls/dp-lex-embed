"""Production wiring: Postgres (psycopg) and Jina (httpx)."""
from __future__ import annotations

import random
import time

import httpx

from dp_lex_embed.actions import Actions
from dp_lex_embed.embeddings import JinaEmbeddingGenerator
from dp_lex_embed.pg_repository import PgInspector, open_run

HTTP_TIMEOUT_SEC = 120.0


def jina_embedder(api_key: str) -> JinaEmbeddingGenerator:
    client = httpx.Client(timeout=HTTP_TIMEOUT_SEC)
    return JinaEmbeddingGenerator(api_key, client, time.sleep, random.Random())  # noqa: S311 - jitter, not security


def build_actions() -> Actions:
    return Actions(open_run, jina_embedder, PgInspector())
