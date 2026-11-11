from __future__ import annotations

import io
import json
from typing import Any

import pytest

import main as entrypoint
import v11
from dp_lex_embed.actions import ActionError, Actions

pytestmark = pytest.mark.unit


def _no_db(_: str) -> Any:
    msg = "no database expected"
    raise AssertionError(msg)


def _no_embedder(_: str) -> Any:
    msg = "no embedder expected"
    raise AssertionError(msg)


def _actions() -> Actions:
    return Actions(_no_db, _no_embedder, None)


def _request(action: str, **params: Any) -> str:
    return json.dumps({"_action": action, "_meta": {"executionId": "e1"}, **params}) + "\n"


def test_should_return_zero_metrics_when_dry_run_has_no_connection_string() -> None:
    result = _actions().run({"dryRun": True, "table": "ds_x.notes"}, None)

    assert result["metrics"]["pending"] == 0
    assert result["metrics"]["embedded"] == 0
    assert result["metrics"]["embed_tokens"] == 0
    assert result["metrics"]["written.ds_x.notes"] == 0


def test_should_answer_check_with_ok_when_dry_run() -> None:
    assert _actions().check({"dryRun": True}) == {"ok": True}


def test_should_raise_config_error_when_table_is_missing() -> None:
    with pytest.raises(ActionError) as ex:
        _actions().run({"dryRun": True}, None)

    assert ex.value.code is v11.ProcErrorCode.CONFIG_ERROR


def test_main_should_emit_one_error_line_for_a_bad_request_and_a_result_for_spec() -> None:
    out = io.StringIO()

    entrypoint.main(io.StringIO("not json\n" + _request("spec")), out, _actions())

    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["error", "result"]
    assert events[0]["code"] == "config_error"
    assert events[1]["data"]["protocol"] == "1.1"
