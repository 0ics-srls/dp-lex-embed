"""Entrypoint of dp-lex-embed: Proc-protocol v1.1 stdin loop (NDJSON in, NDJSON events out on stdout).

Actions: spec, check, run (and terminate / stdin EOF = shutdown). stdout carries protocol events only; diagnostics and tracebacks go to
stderr (logging). A failing action prints exactly one error line with one of the three protocol codes.
"""
from __future__ import annotations

import logging
import sys
from typing import TextIO

import procclass
import v11
from dp_lex_embed.actions import ActionError, Actions

log = logging.getLogger("dp_lex_embed")


def default_actions() -> Actions:
    from dp_lex_embed.wiring import build_actions  # noqa: PLC0415 - heavy imports (psycopg, httpx) only for the real entrypoint

    return build_actions()


def _handle(request: procclass.ProcRequest, actions: Actions, stdout: TextIO) -> None:
    action = request.action
    if action == "spec":
        procclass.emit_result(stdout, actions.spec())
    elif action == "check":
        procclass.emit_result(stdout, actions.check(request.params))
    elif action == "run":
        def progress(percent: int, message: str) -> None:
            procclass.emit_progress(stdout, percent, message)

        progress(0, "starting")
        result = actions.run(request.params, progress)
        progress(100, "done")
        procclass.emit_result(stdout, result)
    else:
        raise ActionError(v11.ProcErrorCode.CONFIG_ERROR, f"unsupported action: {action}")


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, actions: Actions | None = None) -> int:
    handler = actions or default_actions()
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = procclass.parse(line)
        except procclass.ProcProtocolError as ex:
            procclass.emit_error(stdout, v11.ProcErrorCode.CONFIG_ERROR.value, str(ex))
            continue
        if request.action == "terminate":
            break
        try:
            _handle(request, handler, stdout)
        except ActionError as ex:
            log.error("%s failed (%s): %s", request.action, ex.code.value, ex)  # noqa: TRY400 - no traceback needed for a known error
            procclass.emit_error(stdout, ex.code.value, str(ex), v11.error_data(ex.code, ex.retry_after_sec))
        except Exception:
            log.exception("unexpected failure in action %s", request.action)
            procclass.emit_error(stdout, v11.ProcErrorCode.SYSTEM_ERROR.value, f"unexpected failure in action {request.action}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
