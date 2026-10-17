"""Liveness and readiness probes (§7.1)."""

from __future__ import annotations

import sqlite3

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from transcriber.api.deps import get_app_settings, get_conn
from transcriber.api.routes.system import worker_status
from transcriber.config import Settings
from transcriber.store.files import free_disk_gb

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness")
def healthz() -> dict[str, str]:
    """200 while the process is serving requests."""
    return {"status": "ok"}


def _ollama_check(url: str) -> dict[str, object]:
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/version", timeout=2.0)
        resp.raise_for_status()
        return {"ok": True, "version": resp.json().get("version")}
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/readyz", summary="Readiness")
def readyz(
    settings: Settings = Depends(get_app_settings), conn: sqlite3.Connection = Depends(get_conn),
) -> JSONResponse:
    """200 when the DB is writable, disk has room and the worker is alive.

    Ollama is reported but does not affect readiness, since jobs degrade gracefully without it.
    """
    checks: dict[str, dict[str, object]] = {}
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        checks["database"] = {"ok": True}
    except sqlite3.Error as exc:
        checks["database"] = {"ok": False, "error": str(exc)}
    free = free_disk_gb(settings.data_dir)
    checks["disk"] = {"ok": free >= settings.min_free_disk_gb, "free_gb": round(free, 1),
                      "min_free_gb": settings.min_free_disk_gb}
    worker = worker_status(conn)
    checks["worker"] = {"ok": worker["alive"], "last_heartbeat": worker["last_heartbeat"]}
    ready = all(bool(c["ok"]) for c in checks.values())
    checks["ollama"] = _ollama_check(settings.ollama_url)
    return JSONResponse({"status": "ready" if ready else "not_ready", "checks": checks},
                        status_code=200 if ready else 503)
