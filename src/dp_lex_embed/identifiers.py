"""SQL identifiers come from the Source config: validated against a strict pattern and quoted (P78b-01)."""
from __future__ import annotations

import re

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class InvalidIdentifierError(ValueError):
    """The text is not a safe lower-case SQL identifier."""


def check_ident(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name):
        msg = f"invalid identifier {name!r}: expected lower-case letters, digits and underscores"
        raise InvalidIdentifierError(msg)
    return name


def quote_ident(name: str) -> str:
    return f'"{check_ident(name)}"'


def parse_table(value: str) -> tuple[str, str]:
    """`schema.table` into its two validated parts."""
    parts = value.split(".") if isinstance(value, str) else []
    if len(parts) != 2:  # noqa: PLR2004 - schema and table
        msg = f"table must be schema.table, got {value!r}"
        raise InvalidIdentifierError(msg)
    return check_ident(parts[0]), check_ident(parts[1])
