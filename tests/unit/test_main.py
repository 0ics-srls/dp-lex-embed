from __future__ import annotations

import io
import json

import pytest

import main as entrypoint
from dp_lex_embed.actions import Actions

pytestmark = pytest.mark.unit


def test_should_build_the_default_actions_without_network() -> None:
    assert isinstance(entrypoint.default_actions(), Actions)


def test_should_answer_spec_and_a_dry_run_through_the_default_wiring() -> None:
    lines = [
        json.dumps({"_action": "spec", "_meta": {"executionId": "e"}}),
        json.dumps({"_action": "run", "_meta": {"executionId": "e"}, "dryRun": True, "table": "ds_x.notes"}),
        json.dumps({"_action": "check", "_meta": {"executionId": "e"}, "dryRun": True}),
    ]
    out = io.StringIO()

    entrypoint.main(io.StringIO("\n".join(lines) + "\n"), out)

    events = [json.loads(line) for line in out.getvalue().splitlines()]
    results = [e for e in events if e["type"] == "result"]
    assert len(results) == 3
    assert results[1]["data"]["metrics"]["pending"] == 0
    assert results[2]["data"] == {"ok": True}
