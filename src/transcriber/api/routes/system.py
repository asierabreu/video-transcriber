"""GET /v1/system (§7.1)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends

from transcriber import __version__
from transcriber.api.auth import require_api_key
from transcriber.api.deps import get_conn
from transcriber.store import jobs_repo as repo
from transcriber.store.db import parse_iso, utcnow

router = APIRouter(prefix="/v1", tags=["system"], dependencies=[Depends(require_api_key)])

WORKER_ALIVE_S = 60


def worker_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Worker liveness from the heartbeat row."""
    beat = repo.get_system(conn, "worker_heartbeat")
    if beat is None:
        return {"alive": False, "last_heartbeat": None, "current_job": None}
    value, updated_at = beat
    age = (utcnow() - parse_iso(updated_at)).total_seconds()
    return {"alive": age < WORKER_ALIVE_S, "last_heartbeat": updated_at,
            "current_job": value.get("current_job")}


@router.get("/system", summary="Hardware, models, worker and queue status")
def system(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
    """Report what the worker detected at startup and the current queue."""
    profile = repo.get_system(conn, "worker_profile")
    profile_value = profile[0] if profile else {}
    hardware = profile_value.get("hardware", {})
    counts = repo.count_by_status(conn)
    return {
        "version": __version__,
        "hardware": {
            "device": hardware.get("device"),
            "tier": hardware.get("tier"),
            "gpu": hardware.get("gpu_name"),
            "vram_gb": hardware.get("vram_gb"),
            "cpu_threads": hardware.get("cpu_threads"),
            "notes": hardware.get("notes", []),
        },
        "models": profile_value.get("models", {}),
        "worker": worker_status(conn),
        "queue": {"queued": counts["queued"], "running": counts["running"]},
    }
