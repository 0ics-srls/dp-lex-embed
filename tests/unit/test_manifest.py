from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import dp_lex_embed
from dp_lex_embed.actions import Actions

pytestmark = pytest.mark.unit

MANIFEST = json.loads((Path(__file__).resolve().parents[2] / "manifest.json").read_text(encoding="utf-8"))
REQUIRED = {"connString", "table", "keyColumn", "textColumns", "vectorColumn", "tokenBudget", "jinaApiKey"}


def test_should_declare_spec_version_name_and_the_four_actions() -> None:
    assert MANIFEST["specVersion"] == "1.1"
    assert MANIFEST["name"] == "dp-lex-embed"
    assert {"spec", "check", "run", "terminate"} <= set(MANIFEST["actions"])


def test_should_have_one_version_source_for_the_image_tag() -> None:
    assert MANIFEST["version"] == dp_lex_embed.__version__


def test_should_require_the_embed_inputs_in_a_valid_config_schema() -> None:
    config = Actions.spec()["configSchema"]

    jsonschema.Draft202012Validator.check_schema(config)
    assert set(config["required"]) >= REQUIRED
    assert set(config["properties"]) <= set(MANIFEST["actions"]["run"]["input"]["properties"])
