"""Runs one job end to end and records the outcome (§6.2, §6.5)."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from transcriber.backends.base import AsrBackend, DiarizationBackend, LlmBackend
from transcriber.config import Settings
from transcriber.errors import JobCancelled, PipelineError, WorkerStopping
from transcriber.hardware.tiers import ASR_MODEL_CHAIN, AsrAttempt, EffectiveConfig, HardwareProfile
from transcriber.logging import job_id_var
from transcriber.pipeline.render import build_result
from transcriber.pipeline.stages import BackendFactory, JobContext, Stage, run_pipeline
from transcriber.store import jobs_repo as repo
from transcriber.store.files import JobFiles

log = logging.getLogger(__name__)

PROGRESS_WRITE_INTERVAL_S = 2.0
RETRY_BASE_DELAY_S = 30.0


class DefaultBackendFactory:
    """Builds the real faster-whisper, pyannote and Ollama backends."""

    def __init__(self, settings: Settings, eff: EffectiveConfig) -> None:
        self.settings = settings
        self.eff = eff
        self._llm: LlmBackend | None = None

    def available_asr_models(self) -> set[str]:
        from transcriber.backends.asr_faster_whisper import asr_model_available

        names = set(ASR_MODEL_CHAIN) | {self.eff.asr_model}
        return {n for n in names if asr_model_available(n, self.settings.models_dir)}

    def asr(self, attempt: AsrAttempt) -> AsrBackend:
        from transcriber.backends.asr_faster_whisper import FasterWhisperAsr

        return FasterWhisperAsr(
            attempt.model, self.settings.models_dir, attempt.device, attempt.compute_type,
            attempt.batch_size, self.eff.asr_beam_size, self.eff.cpu_threads,
        )

    def diarizer(self, device: str) -> DiarizationBackend:
        from transcriber.backends.diarization_pyannote import PyannoteDiarizer

        return PyannoteDiarizer(self.settings.diarization_model_path, device)

    def llm(self) -> LlmBackend:
        if self._llm is None:
            from transcriber.backends.llm_ollama import OllamaLlm

            self._llm = OllamaLlm(
                self.settings.ollama_url, self.eff.llm_model, self.eff.llm_num_ctx,
                self.settings.llm_timeout_s, self.settings.llm_keep_alive,
            )
        return self._llm


class JobRunner:
    """Executes claimed jobs."""

    def __init__(
        self, settings: Settings, conn: sqlite3.Connection, profile: HardwareProfile,
        eff: EffectiveConfig, factory: BackendFactory, stop_event: threading.Event,
        stages: list[Stage] | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.profile = profile
        self.eff = eff
        self.factory = factory
        self.stop_event = stop_event
        self.stages = stages
        self.current_job_id: str | None = None

    def _make_context(self, job: dict[str, Any], files: JobFiles) -> JobContext:
        job_id = job["id"]
        last_time = 0.0
        last_stage = ""

        def on_progress(stage: str, overall: float) -> None:
            nonlocal last_time, last_stage
            now = time.monotonic()
            if stage != last_stage or now - last_time >= PROGRESS_WRITE_INTERVAL_S:
                repo.update_progress(self.conn, job_id, stage, overall)
                last_time, last_stage = now, stage

        def check_cancel() -> None:
            if self.stop_event.is_set():
                raise WorkerStopping()
            if repo.is_cancel_requested(self.conn, job_id):
                raise JobCancelled()

        meta: dict[str, Any] = files.read_json("meta.json") if files.exists("meta.json") else {}
        meta.update({"effective": self.eff.to_dict(), "hardware": self.profile.to_dict(),
                     "attempt": job["attempts"]})
        return JobContext(
            job_id=job_id, files=files, options=job["options"], media=job["media"],
            eff=self.eff, settings=self.settings, factory=self.factory,
            attempts=job["attempts"], warnings=list(job["warnings"]), meta=meta,
            on_warning=lambda msg: repo.add_warnings(self.conn, job_id, [msg]),
            on_progress=on_progress, check_cancel=check_cancel,
        )

    def run(self, job: dict[str, Any]) -> str:
        """Process one claimed job; returns the final status for this attempt."""
        job_id = job["id"]
        token = job_id_var.set(job_id)
        self.current_job_id = job_id
        files = JobFiles(self.settings.jobs_dir, job_id)
        started = time.monotonic()
        try:
            ctx = self._make_context(job, files)
            files.write_json("meta.json", ctx.meta)
            log.info("job started", extra={"attempt": job["attempts"]})
            run_pipeline(ctx, self.stages)
            result = build_result(files)
            result["processing_s"] = round(time.monotonic() - started, 1)
            result["stage_durations"] = ctx.meta.get("stage_durations", {})
            files.write_json("meta.json", {**ctx.meta, "result": result})
            if not self.settings.keep_media:
                files.delete_media()
            repo.mark_succeeded(self.conn, job_id, result)
            log.info("job succeeded", extra={"processing_s": result["processing_s"]})
            return "succeeded"
        except JobCancelled:
            repo.mark_cancelled(self.conn, job_id)
            files.delete_all()
            log.info("job cancelled")
            return "cancelled"
        except WorkerStopping:
            repo.requeue(self.conn, job_id, 0, refund_attempt=True)
            log.info("worker stopping; job returned to the queue")
            return "queued"
        except PipelineError as exc:
            return self._handle_failure(job, exc.code, str(exc), exc.transient)
        except Exception as exc:  # noqa: BLE001 - unexpected errors: transient once, then permanent
            log.exception("unexpected error while processing job")
            return self._handle_failure(job, "internal_error", f"{type(exc).__name__}: {exc}",
                                        transient=job["attempts"] <= 1)
        finally:
            self.current_job_id = None
            job_id_var.reset(token)

    def _handle_failure(self, job: dict[str, Any], code: str, message: str,
                        transient: bool) -> str:
        attempts = job["attempts"]
        if transient and attempts < self.settings.job_max_attempts:
            delay = RETRY_BASE_DELAY_S * 2 ** attempts
            repo.requeue(self.conn, job["id"], delay)
            log.warning("job failed, will retry", extra={"code": code, "error": message,
                                                         "retry_in_s": delay})
            return "queued"
        repo.mark_failed(self.conn, job["id"], code, message)
        log.error("job failed", extra={"code": code, "error": message})
        return "failed"


def required_models_missing(settings: Settings, eff: EffectiveConfig) -> list[str]:
    """Describe models the worker needs but cannot find locally."""
    from transcriber.backends.asr_faster_whisper import asr_model_available
    from transcriber.backends.diarization_pyannote import diarization_model_available

    missing = []
    if not asr_model_available(eff.asr_model, settings.models_dir):
        missing.append(f"ASR model {eff.asr_model!r} (expected in "
                       f"{Path(settings.models_dir) / 'whisper' / eff.asr_model})")
    if not diarization_model_available(settings.diarization_model_path):
        missing.append(f"diarization pipeline (expected in {settings.diarization_model_path})")
    return missing
