"""FastAPI dependencies."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Request

from transcriber.config import Settings
from transcriber.store.db import connect


def get_app_settings(request: Request) -> Settings:
    """Settings bound to the running app (set in create_app)."""
    settings: Settings = request.app.state.settings
    return settings


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    """A short-lived SQLite connection per request."""
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()
