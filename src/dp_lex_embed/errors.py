"""Errors shared across the layers of the embed step."""
from __future__ import annotations


class RepositoryError(Exception):
    """A database operation failed; it aborts the run (batches already stored stay stored)."""


class DbConnectError(Exception):
    """The dataset database cannot be reached (message names the role only, never the connection string).

    `sqlstate` is the server's SQLSTATE when it answered (class 28 = authentication / authorization), None when it did not.
    """

    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate

    @property
    def is_login_rejected(self) -> bool:
        return self.sqlstate is not None and self.sqlstate.startswith("28")
