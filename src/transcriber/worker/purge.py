"""Retention purge and temp-file cleanup (§10 file cleanup)."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path

from transcriber.config import Settings
from transcriber.ids import is_valid_job_id
from transcriber.store import jobs_repo as repo
from transcriber.store.files import JobFiles

log = logging.getLogger(__name__)

STALE_TEMP_S = 24 * 3600


def purge_expired(conn: sqlite3.Connection, settings: Settings) -> int:
    """Delete finished jobs older than RETENTION_DAYS, files first. Returns the count."""
    ids = repo.expired_job_ids(conn, settings.retention_days)
    for job_id in ids:
        JobFiles(settings.jobs_dir, job_id).delete_all()
        repo.delete_job(conn, job_id)
    if ids:
        log.info("purged expired jobs", extra={"count": len(ids)})
    return len(ids)


def cleanup_stale_files(conn: sqlite3.Connection, jobs_dir: Path,
                        max_age_s: float = STALE_TEMP_S) -> int:
    """Remove temp files and orphaned job directories older than `max_age_s`."""
    if not jobs_dir.exists():
        return 0
    cutoff = time.time() - max_age_s
    known = repo.all_job_ids(conn)
    removed = 0
    for entry in jobs_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            old = entry.stat().st_mtime < cutoff
        except FileNotFoundError:
            continue
        if old and (not is_valid_job_id(entry.name) or entry.name not in known):
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
            continue
        for tmp in list(entry.glob("*.tmp")) + list(entry.glob("*.tmp.wav")) + \
                list(entry.glob("upload.part")) + list(entry.glob("out/*.tmp")):
            try:
                if tmp.stat().st_mtime < cutoff:
                    tmp.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
    if removed:
        log.info("removed stale files", extra={"count": removed})
    return removed
