"""Worker entry point: `transcriber-worker`."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from types import FrameType

from transcriber import __version__
from transcriber.config import get_settings
from transcriber.hardware.detect import detect_hardware
from transcriber.hardware.tiers import resolve_effective
from transcriber.logging import setup_logging
from transcriber.store import jobs_repo as repo
from transcriber.store.db import open_db
from transcriber.worker.heartbeat import Heartbeat, worker_identity
from transcriber.worker.purge import cleanup_stale_files, purge_expired
from transcriber.worker.runner import DefaultBackendFactory, JobRunner, required_models_missing

log = logging.getLogger("transcriber.worker")

STALE_JOB_S = 300.0
RECOVER_INTERVAL_S = 60.0
PURGE_INTERVAL_S = 24 * 3600.0


def main() -> None:
    """Detect hardware, verify models, then process jobs until SIGTERM/SIGINT."""
    settings = get_settings()
    settings.apply_library_env()
    setup_logging(settings.log_level, settings.log_format)

    conn = open_db(settings.db_path)
    profile = detect_hardware(settings)
    eff = resolve_effective(settings, profile)
    missing = required_models_missing(settings, eff)
    if missing:
        for item in missing:
            log.error("required model missing: %s", item)
        log.error("download models first: `make models` (Docker) or `make models-local` "
                  "(native), e.g. `make models TIER=%s`", eff.tier)
        sys.exit(2)

    worker_id = worker_identity()
    repo.set_system(conn, "worker_profile", {
        "version": __version__, "worker_id": worker_id, "hardware": profile.to_dict(),
        "models": eff.to_dict(), "ollama_url": settings.ollama_url,
    })
    log.info("worker starting", extra={"worker_id": worker_id, "effective": eff.to_dict()})

    stop_event = threading.Event()

    def _stop(signum: int, _frame: FrameType | None) -> None:
        log.info("received signal %s, stopping after the current stage", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    runner = JobRunner(settings, conn, profile, eff, DefaultBackendFactory(settings, eff),
                       stop_event)
    heartbeat = Heartbeat(settings.db_path, worker_id, lambda: runner.current_job_id, stop_event)
    heartbeat.start()

    last_recover = 0.0
    last_purge = 0.0
    while not stop_event.is_set():
        now = time.monotonic()
        if now - last_recover >= RECOVER_INTERVAL_S:
            recovered = repo.recover_stale(conn, STALE_JOB_S, settings.job_max_attempts)
            if recovered:
                log.warning("recovered stale jobs", extra={"job_ids": recovered})
            last_recover = now
        if now - last_purge >= PURGE_INTERVAL_S:
            purge_expired(conn, settings)
            cleanup_stale_files(conn, settings.jobs_dir)
            last_purge = now
        job = repo.claim_next(conn, worker_id)
        if job is None:
            stop_event.wait(settings.worker_poll_interval_s)
            continue
        runner.run(job)

    heartbeat.join(timeout=5)
    conn.close()
    log.info("worker stopped")


if __name__ == "__main__":
    main()
