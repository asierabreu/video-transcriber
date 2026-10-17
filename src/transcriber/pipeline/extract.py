"""Stage 3: extract the first audio track as 16 kHz mono 16-bit PCM WAV."""

from __future__ import annotations

import os
import subprocess

from transcriber.errors import MediaError
from transcriber.pipeline.stages import JobContext

EXTRACT_TIMEOUT_S = 3600


def run(ctx: JobContext) -> None:
    """Run ffmpeg on the stored source file and write audio.wav atomically."""
    source = ctx.files.source()
    if source is None:
        raise MediaError("source file is missing from the job directory")
    tmp = ctx.files.path("audio.tmp.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-f", "wav", str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=EXTRACT_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise MediaError("audio extraction timed out") from exc
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        stderr = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise MediaError(f"ffmpeg failed to extract audio: {stderr or 'no audio produced'}")
    os.replace(tmp, ctx.files.audio)
    ctx.progress(1.0)
