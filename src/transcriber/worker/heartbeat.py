"""Background thread that keeps the worker and current-job heartbeats fresh (§6.1)."""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from transcriber.store import jobs_repo as repo
from transcriber.store.db import connect

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 15.0


def worker_identity() -> str:
    """Stable-enough identifier for this worker process."""
    return f"{socket.gethostname()}-{os.getpid()}"


class Heartbeat(threading.Thread):
    """Writes `worker_heartbeat` to the system table and touches the running job."""

    def __init__(self, db_path: Path, worker_id: str, current_job: Callable[[], str | None],
                 stop_event: threading.Event, interval_s: float = HEARTBEAT_INTERVAL_S) -> None:
        super().__init__(name="heartbeat", daemon=True)
        self.db_path = db_path
        self.worker_id = worker_id
        self.current_job = current_job
        self.stop_event = stop_event
        self.interval_s = interval_s

    def beat(self, conn: sqlite3.Connection) -> None:
        """Write one heartbeat."""
        job_id = self.current_job()
        repo.set_system(conn, "worker_heartbeat",
                        {"worker_id": self.worker_id, "current_job": job_id})
        if job_id:
            repo.touch_heartbeat(conn, job_id)

    def run(self) -> None:
        conn = connect(self.db_path)
        try:
            while True:
                try:
                    self.beat(conn)
                except Exception:  # noqa: BLE001 - never let the heartbeat thread die
                    log.exception("heartbeat failed")
                if self.stop_event.wait(self.interval_s):
                    break
        finally:
            conn.close()
