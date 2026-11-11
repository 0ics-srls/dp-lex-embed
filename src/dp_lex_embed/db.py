"""Connection helpers: the Host injects an Npgsql-style connection string, psycopg wants libpq keyword/value format.

`to_libpq` / `role_of` are copied from dp-sk-zakony-ingest (independent deployables, no shared package).
The connection string carries the password: it never goes into a log line or an exception message.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psycopg

from dp_lex_embed.errors import DbConnectError

if TYPE_CHECKING:
    from psycopg import Connection

log = logging.getLogger(__name__)

_NPGSQL_TO_LIBPQ_KEY = {"host": "host", "port": "port", "database": "dbname", "username": "user", "password": "password"}


def to_libpq(conn_string: str) -> str:
    """Converts a .NET Npgsql-style connection string into libpq keyword/value format; a string without ';' passes through."""
    if ";" not in conn_string:
        return conn_string
    parts = []
    for raw_chunk in conn_string.split(";"):
        chunk = raw_chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        parts.append(f"{_NPGSQL_TO_LIBPQ_KEY.get(key.strip().lower(), key.strip().lower())}={value.strip()}")
    return " ".join(parts)


def role_of(conn_string: str) -> str:
    """The database user of a connection string (never the password), 'unknown' when absent."""
    for token in to_libpq(conn_string).split():
        if token.startswith("user="):
            return token[len("user="):] or "unknown"
    return "unknown"


def connect(conn_string: str, *, autocommit: bool = False) -> Connection:
    """Opens a connection with the credentials of the dataset role (the table is always schema-qualified).

    With `autocommit=True` the caller controls transactions explicitly (`connection.transaction()` is then a real BEGIN / COMMIT,
    not a savepoint inside an implicit transaction). Failures name the role and the error class only.
    """
    try:
        connection = psycopg.connect(to_libpq(conn_string), autocommit=False)
    except psycopg.Error as ex:
        role = role_of(conn_string)
        log.error("cannot connect to the dataset database as %s (%s)", role, type(ex).__name__)  # noqa: TRY400 - a traceback would carry the DSN
        msg = f"cannot connect to the dataset database as role {role} ({type(ex).__name__})"
        raise DbConnectError(msg, sqlstate=ex.sqlstate) from None
    connection.commit()
    connection.autocommit = autocommit
    return connection


