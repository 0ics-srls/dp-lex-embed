from __future__ import annotations

import pytest

from dp_lex_embed.identifiers import InvalidIdentifierError, parse_table, quote_ident

pytestmark = pytest.mark.unit


def test_should_split_a_schema_qualified_table() -> None:
    assert parse_table("ds_x.notes") == ("ds_x", "notes")


@pytest.mark.parametrize("bad", ["x; drop", "DS_X.notes", "", "a.b.c", "notes", "ds_x.", '"a".b', "ds_x.no tes"])
def test_should_reject_hostile_or_unqualified_tables(bad: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        parse_table(bad)


def test_should_quote_a_valid_column_and_reject_an_invalid_one() -> None:
    assert quote_ident("body") == '"body"'
    with pytest.raises(InvalidIdentifierError):
        quote_ident("body; --")
