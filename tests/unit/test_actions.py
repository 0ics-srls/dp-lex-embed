from __future__ import annotations

import io
import json
from typing import Any

import pytest

import main as entrypoint
import v11
from dp_lex_embed.actions import ActionError, Actions

pytestmark = pytest.mark.unit


def _no_db(*_: Any) -> Any:
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


def test_should_report_exact_metric_keys_and_written_equal_embedded_when_run() -> None:
    from builders import FakeEmbedder, FakeRepository, FakeRunOpener  # noqa: PLC0415

    repo = FakeRepository({1: "aaaaaa", 2: "bbbbbb", 3: "  "})
    actions = Actions(FakeRunOpener(repo), lambda _: FakeEmbedder(), None)
    params = {"connString": "c", "jinaApiKey": "k", "table": "ds_x.notes", "keyColumn": "id", "textColumns": ["body"],
              "vectorColumn": "embedding", "tokenBudget": 100, "batchSize": 2}

    metrics = actions.run(params, None)["metrics"]

    assert set(metrics) == {"pending", "embedded", "skipped", "embed_tokens", "written.ds_x.notes"}
    assert metrics["embedded"] == 2
    assert metrics["written.ds_x.notes"] == 2
    assert metrics["pending"] == 3
    assert metrics["skipped"] == 1
    assert metrics["embed_tokens"] == 4


def test_should_plan_without_embedding_when_dry_run_has_a_connection() -> None:
    from builders import FakeEmbedder, FakeRepository, FakeRunOpener  # noqa: PLC0415

    repo = FakeRepository({1: "aaaaaa", 2: "bbbbbb"})
    embedder = FakeEmbedder()
    actions = Actions(FakeRunOpener(repo), lambda _: embedder, None)
    params = {"dryRun": True, "connString": "c", "table": "ds_x.notes", "keyColumn": "id", "textColumns": ["body"],
              "vectorColumn": "embedding", "tokenBudget": 100}

    metrics = actions.run(params, None)["metrics"]

    assert metrics["pending"] == 2
    assert metrics["embed_tokens"] == 4
    assert metrics["embedded"] == 0
    assert embedder.calls == []
    assert repo.vectors == {}
