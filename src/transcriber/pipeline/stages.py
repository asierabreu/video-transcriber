"""Job context, backend factory interface, and the ordered stage runner (§3, §6.2)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from transcriber.backends.base import AsrBackend, DiarizationBackend, LlmBackend
from transcriber.config import Settings
from transcriber.hardware.tiers import AsrAttempt, EffectiveConfig
from transcriber.logging import stage_var
from transcriber.store.files import JobFiles

log = logging.getLogger(__name__)


class BackendFactory(Protocol):
    """Creates backends for one job. The worker supplies the real one; tests supply fakes."""

    def available_asr_models(self) -> set[str]:
        """Model names that can be loaded offline."""
        ...

    def asr(self, attempt: AsrAttempt) -> AsrBackend:
        """Load an ASR backend for the given configuration."""
        ...

    def diarizer(self, device: str) -> DiarizationBackend:
        """Load the diarization backend on `device`."""
        ...

    def llm(self) -> LlmBackend:
        """Return the LLM backend."""
        ...


@dataclass
class JobContext:
    """Everything a stage needs to process one job."""

    job_id: str
    files: JobFiles
    options: dict[str, Any]
    media: dict[str, Any]
    eff: EffectiveConfig
    settings: Settings
    factory: BackendFactory
    attempts: int
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    on_warning: Callable[[str], None] = lambda _msg: None
    on_progress: Callable[[str, float], None] = lambda _stage, _p: None
    check_cancel: Callable[[], None] = lambda: None
    _stage: str = ""
    _stage_base: float = 0.0
    _stage_weight: float = 0.0

    @property
    def last_attempt(self) -> bool:
        """True when a transient failure would no longer be retried."""
        return self.attempts >= self.settings.job_max_attempts

    def warn(self, message: str) -> None:
        """Record a client-visible warning (§5.3) and persist it."""
        if message not in self.warnings:
            self.warnings.append(message)
            log.warning(message)
            self.on_warning(message)

    def progress(self, fraction: float) -> None:
        """Report progress within the current stage (0..1)."""
        overall = self._stage_base + self._stage_weight * min(max(fraction, 0.0), 1.0)
        self.on_progress(self._stage, overall)


StageFn = Callable[[JobContext], None]


@dataclass(frozen=True)
class Stage:
    """A pipeline stage, its progress weight and the artifact that marks it complete."""

    name: str
    weight: float
    artifact: str
    run: StageFn


def build_stages(options: dict[str, Any] | None = None) -> list[Stage]:
    """Build the shortest stage list that satisfies the requested job options."""
    from transcriber.pipeline import align, cleanup, diarize, extract, render, summarize, transcribe

    options = options or {}
    selected = [
        Stage("extracting", 0.05, "audio.wav", extract.run),
        Stage("transcribing", 0.45, "asr.json", transcribe.run),
    ]
    if options.get("diarization"):
        selected.append(Stage("diarizing", 0.20, "diarization.json", diarize.run))
    if options.get("diarization") or options.get("cleanup") or options.get("summary"):
        selected.append(Stage("aligning", 0.02, "aligned.json", align.run))
    if options.get("cleanup") or options.get("summary"):
        selected.append(Stage("cleaning", 0.18, "cleaned.json", cleanup.run))
    if options.get("summary"):
        selected.append(Stage("summarizing", 0.08, "summary.json", summarize.run))
    selected.append(
        Stage("rendering", 0.02, "out/transcript.json", render.run),
    )
    return selected


def run_pipeline(ctx: JobContext, stages: list[Stage] | None = None) -> None:
    """Run every stage in order, skipping stages whose artifact already exists (resume)."""
    stages = stages or build_stages(ctx.options)
    durations: dict[str, float] = ctx.meta.setdefault("stage_durations", {})
    total_weight = sum(stage.weight for stage in stages)
    base = 0.0
    for stage in stages:
        ctx.check_cancel()
        weight = stage.weight / total_weight
        ctx._stage, ctx._stage_base, ctx._stage_weight = stage.name, base, weight
        token = stage_var.set(stage.name)
        try:
            if ctx.files.exists(stage.artifact):
                log.info("stage already complete, skipping", extra={"artifact": stage.artifact})
            else:
                ctx.progress(0.0)
                started = time.monotonic()
                log.info("stage started")
                stage.run(ctx)
                durations[stage.name] = round(time.monotonic() - started, 2)
                log.info("stage finished", extra={"duration_s": durations[stage.name]})
            base += weight
            ctx.on_progress(stage.name, base)
        finally:
            stage_var.reset(token)
