"""Job queue and metadata persistence (the `jobs` and `system` tables)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from transcriber.store.db import now_iso

JSON_FIELDS = ("options", "media", "result", "warnings", "error")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")


def _row_to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    job = dict(row)
    for field in JSON_FIELDS:
        if job.get(field) is not None:
            job[field] = json.loads(job[field])
    job["cancel_requested"] = bool(job["cancel_requested"])
    return job


def create_job(
    conn: sqlite3.Connection, job_id: str, title: str, options: dict[str, Any],
    media: dict[str, Any],
) -> dict[str, Any]:
    """Insert a new queued job and return it."""
    conn.execute(
        "INSERT INTO jobs (id, title, status, progress, options, media, warnings, created_at) "
        "VALUES (?, ?, 'queued', 0, ?, ?, '[]', ?)",
        (job_id, title, json.dumps(options), json.dumps(media), now_iso()),
    )
    job = get_job(conn, job_id)
    assert job is not None
    return job


def get_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """Fetch one job by ID."""
    return _row_to_job(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def list_jobs(
    conn: sqlite3.Connection, status: str | None, limit: int, cursor: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    """List jobs newest first. The cursor is the last ID of the previous page."""
    sql = "SELECT * FROM jobs WHERE 1=1"
    args: list[Any] = []
    if status:
        sql += " AND status = ?"
        args.append(status)
    if cursor:
        sql += " AND id < ?"
        args.append(cursor)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit + 1)
    rows = [j for j in (_row_to_job(r) for r in conn.execute(sql, args)) if j]
    next_cursor = rows[limit - 1]["id"] if len(rows) > limit else None
    return rows[:limit], next_cursor


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    """Number of jobs in each status."""
    counts = {s: 0 for s in ("queued", "running", *TERMINAL_STATUSES)}
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
        counts[row["status"]] = row["n"]
    return counts


def queue_position(conn: sqlite3.Connection, job: dict[str, Any]) -> int | None:
    """1-based position among queued jobs, or None if the job is not queued."""
    if job["status"] != "queued":
        return None
    row = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND (created_at < ? OR "
        "(created_at = ? AND id < ?))",
        (job["created_at"], job["created_at"], job["id"]),
    ).fetchone()
    return int(row[0]) + 1


def claim_next(conn: sqlite3.Connection, worker_id: str) -> dict[str, Any] | None:
    """Atomically claim the oldest runnable queued job (§6.1)."""
    now = now_iso()
    row = conn.execute(
        "UPDATE jobs SET status = 'running', worker_id = ?, heartbeat_at = ?, "
        "started_at = COALESCE(started_at, ?), attempts = attempts + 1 "
        "WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 "
        "AND (not_before IS NULL OR not_before <= ?) ORDER BY created_at, id LIMIT 1) "
        "RETURNING *",
        (worker_id, now, now, now),
    ).fetchone()
    return _row_to_job(row)


def update_progress(conn: sqlite3.Connection, job_id: str, stage: str, progress: float) -> None:
    """Record the current stage and overall progress."""
    conn.execute(
        "UPDATE jobs SET stage = ?, progress = ?, heartbeat_at = ? WHERE id = ?",
        (stage, round(min(max(progress, 0.0), 1.0), 4), now_iso(), job_id),
    )


def touch_heartbeat(conn: sqlite3.Connection, job_id: str) -> None:
    """Refresh the running job's heartbeat."""
    conn.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (now_iso(), job_id))


def add_warnings(conn: sqlite3.Connection, job_id: str, warnings: list[str]) -> None:
    """Append warnings, skipping duplicates."""
    if not warnings:
        return
    job = get_job(conn, job_id)
    if job is None:
        return
    merged = list(job["warnings"])
    for w in warnings:
        if w not in merged:
            merged.append(w)
    conn.execute("UPDATE jobs SET warnings = ? WHERE id = ?", (json.dumps(merged), job_id))


def mark_succeeded(conn: sqlite3.Connection, job_id: str, result: dict[str, Any]) -> None:
    """Finish a job successfully."""
    conn.execute(
        "UPDATE jobs SET status = 'succeeded', stage = NULL, progress = 1, result = ?, "
        "error = NULL, finished_at = ? WHERE id = ?",
        (json.dumps(result), now_iso(), job_id),
    )


def mark_failed(conn: sqlite3.Connection, job_id: str, code: str, message: str) -> None:
    """Finish a job with an error."""
    error = {"code": code, "message": message, "details": {}}
    conn.execute(
        "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
        (json.dumps(error), now_iso(), job_id),
    )


def mark_cancelled(conn: sqlite3.Connection, job_id: str) -> None:
    """Finish a job as cancelled."""
    error = {"code": "cancelled", "message": "Job was cancelled.", "details": {}}
    conn.execute(
        "UPDATE jobs SET status = 'cancelled', stage = NULL, error = ?, finished_at = ? "
        "WHERE id = ?",
        (json.dumps(error), now_iso(), job_id),
    )


def cancel_if_queued(conn: sqlite3.Connection, job_id: str) -> bool:
    """Cancel a job only if it is still queued; returns True if it was cancelled."""
    error = {"code": "cancelled", "message": "Job was cancelled.", "details": {}}
    cur = conn.execute(
        "UPDATE jobs SET status = 'cancelled', cancel_requested = 1, error = ?, finished_at = ? "
        "WHERE id = ? AND status = 'queued'",
        (json.dumps(error), now_iso(), job_id),
    )
    return cur.rowcount == 1


def requeue(conn: sqlite3.Connection, job_id: str, delay_s: float, *,
            refund_attempt: bool = False) -> None:
    """Return a running job to the queue, optionally after a delay."""
    conn.execute(
        "UPDATE jobs SET status = 'queued', worker_id = NULL, not_before = ?, "
        "attempts = attempts - ? WHERE id = ?",
        (now_iso(delay_s) if delay_s > 0 else None, 1 if refund_attempt else 0, job_id),
    )


def request_cancel(conn: sqlite3.Connection, job_id: str) -> None:
    """Flag a job for cancellation; the worker acts on it between stages and chunks."""
    conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))


def is_cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    """True if the job was flagged for cancellation or deleted."""
    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row is None or bool(row[0])


def delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    """Delete the job row."""
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def recover_stale(conn: sqlite3.Connection, stale_after_s: float, max_attempts: int) -> list[str]:
    """Requeue running jobs whose heartbeat is older than `stale_after_s` (§6.1).

    Jobs that already used all attempts are failed instead. Returns affected job IDs.
    """
    cutoff = now_iso(-stale_after_s)
    rows = conn.execute(
        "SELECT id, attempts FROM jobs WHERE status = 'running' AND "
        "(heartbeat_at IS NULL OR heartbeat_at < ?)",
        (cutoff,),
    ).fetchall()
    for row in rows:
        if row["attempts"] >= max_attempts:
            mark_failed(conn, row["id"], "out_of_resources",
                        "Worker stopped while processing this job too many times.")
        else:
            requeue(conn, row["id"], 0)
    return [r["id"] for r in rows]


def expired_job_ids(conn: sqlite3.Connection, retention_days: int) -> list[str]:
    """Finished jobs older than the retention period."""
    if retention_days <= 0:
        return []
    cutoff = now_iso(-retention_days * 86400)
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status IN ('succeeded','failed','cancelled') "
        "AND finished_at IS NOT NULL AND finished_at < ?",
        (cutoff,),
    )
    return [r["id"] for r in rows]


def all_job_ids(conn: sqlite3.Connection) -> set[str]:
    """Every job ID in the database."""
    return {r["id"] for r in conn.execute("SELECT id FROM jobs")}


def finished_jobs_since(conn: sqlite3.Connection, since_iso: str) -> list[dict[str, Any]]:
    """Succeeded jobs finished after `since_iso` (for metrics)."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status = 'succeeded' AND finished_at >= ? ORDER BY finished_at",
        (since_iso,),
    )
    return [j for j in (_row_to_job(r) for r in rows) if j]


def total_llm_fallbacks(conn: sqlite3.Connection) -> int:
    """Sum of LLM fallbacks recorded in succeeded jobs' results."""
    row = conn.execute(
        "SELECT COALESCE(SUM(json_extract(result, '$.llm_fallbacks')), 0) FROM jobs "
        "WHERE status = 'succeeded'"
    ).fetchone()
    return int(row[0] or 0)


def set_system(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Upsert a value in the `system` table."""
    conn.execute(
        "INSERT INTO system (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(value), now_iso()),
    )


def get_system(conn: sqlite3.Connection, key: str) -> tuple[Any, str] | None:
    """Return (value, updated_at) for a `system` key, or None."""
    row = conn.execute("SELECT value, updated_at FROM system WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return json.loads(row["value"]), row["updated_at"]
