"""Stage 4: speech recognition with the §5.3 fallback chain."""

from __future__ import annotations

import logging

from transcriber.backends.base import AsrResult
from transcriber.errors import ModelLoadError, ResourceExhaustedError
from transcriber.hardware.tiers import AsrAttempt, asr_fallback_chain
from transcriber.pipeline.stages import JobContext

log = logging.getLogger(__name__)


def _attempt(ctx: JobContext, attempt: AsrAttempt) -> AsrResult:
    backend = ctx.factory.asr(attempt)
    try:
        return backend.transcribe(ctx.files.audio, ctx.progress)
    finally:
        backend.close()


def run(ctx: JobContext) -> None:
    """Transcribe audio.wav into asr.json, falling back on GPU memory/load failures."""
    chain = asr_fallback_chain(ctx.eff, ctx.factory.available_asr_models())
    last_error: Exception | None = None
    for attempt in chain:
        ctx.check_cancel()
        if attempt.reason:
            ctx.warn(f"transcription: {attempt.reason} after error: {last_error}")
        try:
            result = _attempt(ctx, attempt)
        except ModelLoadError:
            raise
        except (RuntimeError, MemoryError) as exc:
            # CTranslate2 and CUDA report OOM and driver problems as RuntimeError.
            log.warning("ASR attempt failed", extra={"attempt": attempt.__dict__,
                                                     "error": str(exc)})
            last_error = exc
            continue
        ctx.meta["asr"] = {"model": attempt.model, "compute_type": attempt.compute_type,
                           "device": attempt.device, "batch_size": attempt.batch_size}
        ctx.files.write_json("asr.json", {**result, "model": attempt.model})
        return
    raise ResourceExhaustedError(f"transcription failed on every fallback: {last_error}")
