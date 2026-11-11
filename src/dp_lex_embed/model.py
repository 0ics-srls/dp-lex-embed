"""Value types of the embed step."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "jina-embeddings-v3"
DEFAULT_BATCH_SIZE = 16
CHARS_PER_TOKEN = 3


@dataclass(frozen=True, slots=True)
class Row:
    key: object
    text: str


@dataclass(frozen=True, slots=True)
class EmbedResult:
    vectors: list[list[float]]
    tokens: int


@dataclass(frozen=True, slots=True)
class EmbedOptions:
    table: str
    key_column: str
    text_columns: tuple[str, ...]
    vector_column: str
    token_budget: int
    batch_size: int = DEFAULT_BATCH_SIZE
    model: str = DEFAULT_MODEL


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)
