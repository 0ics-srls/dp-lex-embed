"""Production wiring (Postgres + Jina); filled in block 04."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dp_lex_embed.actions import Actions


def build_actions() -> Actions:
    msg = "production wiring is added with the Postgres repository"
    raise NotImplementedError(msg)
