"""Stage 5: speaker diarization, falling back to CPU if the accelerator fails."""

from __future__ import annotations

import logging

from transcriber.backends.base import SpeakerTurn
from transcriber.errors import ModelLoadError, PipelineError, ResourceExhaustedError
from transcriber.pipeline.stages import JobContext

log = logging.getLogger(__name__)


def run(ctx: JobContext) -> None:
    """Diarize audio.wav into diarization.json."""
    asr = ctx.files.read_json("asr.json")
    if not asr["words"]:
        ctx.files.write_json("diarization.json", {"turns": [], "device": None})
        return
    devices = [ctx.eff.diarization_device]
    if ctx.eff.diarization_device != "cpu":
        devices.append("cpu")
    last_error: Exception | None = None
    for device in devices:
        ctx.check_cancel()
        if last_error is not None:
            ctx.warn(f"diarization: falling back to {device} after error: {last_error}")
        try:
            backend = ctx.factory.diarizer(device)
        except ModelLoadError:
            raise
        except (RuntimeError, MemoryError) as exc:
            last_error = exc
            continue
        try:
            turns: list[SpeakerTurn] = backend.diarize(
                ctx.files.audio,
                ctx.options.get("num_speakers"),
                ctx.options.get("min_speakers"),
                ctx.options.get("max_speakers"),
            )
        except (RuntimeError, MemoryError) as exc:
            log.warning("diarization failed", extra={"device": device, "error": str(exc)})
            last_error = exc
            continue
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a permanent, coded failure
            raise PipelineError(f"diarization failed: {exc}", code="diarization_failed") from exc
        finally:
            backend.close()
        ctx.meta["diarization"] = {"device": device}
        ctx.files.write_json("diarization.json", {"turns": turns, "device": device})
        ctx.progress(1.0)
        return
    raise ResourceExhaustedError(f"diarization failed on every device: {last_error}")
