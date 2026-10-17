"""SQLite connection handling and schema migrations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path


def utcnow() -> datetime:
    """Current time in UTC."""
    return datetime.now(UTC)


def to_iso(value: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with a trailing Z (second precision)."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso(offset_s: float = 0) -> str:
    """ISO-8601 UTC timestamp for now plus `offset_s` seconds."""
    return to_iso(utcnow() + timedelta(seconds=offset_s))


def parse_iso(value: str) -> datetime:
    """Parse a timestamp written by `to_iso`."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection configured for concurrent API reads and worker writes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migration_files() -> list[tuple[int, str]]:
    folder = resources.files("transcriber.store") / "migrations"
    items: list[tuple[int, str]] = []
    for entry in folder.iterdir():
        name = entry.name
        if name.endswith(".sql") and name[:4].isdigit():
            items.append((int(name[:4]), entry.read_text(encoding="utf-8")))
    return sorted(items)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations; returns the resulting schema version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, sql in _migration_files():
        if version > current:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _exec_script(conn, sql)
                conn.execute(f"PRAGMA user_version = {version}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            current = version
    return int(current)


def _exec_script(conn: sqlite3.Connection, sql: str) -> None:
    # executescript() would COMMIT the open transaction, so run statements one by one.
    for statement in sql.split(";"):
        if statement.strip():
            conn.execute(statement)


def open_db(db_path: Path) -> sqlite3.Connection:
    """Connect and migrate."""
    conn = connect(db_path)
    migrate(conn)
    return conn
