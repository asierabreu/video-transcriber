"""ffprobe-based media validation (stage 2, run by the API before a job is queued)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from transcriber.errors import MediaValidationError

FFPROBE_TIMEOUT_S = 120


def run_ffprobe(path: Path) -> dict[str, Any]:
    """Run ffprobe and return its JSON output. Raises MediaValidationError if unreadable."""
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=FFPROBE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaValidationError(
            422, "unreadable_media", "Timed out while reading the file."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not installed or not on PATH") from exc
    if proc.returncode != 0:
        raise MediaValidationError(
            422, "unreadable_media", "The file could not be read as a video.",
            {"ffprobe": proc.stderr.decode("utf-8", "replace")[-300:].strip()},
        )
    try:
        data: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaValidationError(422, "unreadable_media", "Unexpected ffprobe output.") from exc
    return data


def detect_container(probe: dict[str, Any]) -> tuple[str | None, str]:
    """Return (container name or None, raw ffprobe format_name)."""
    fmt = probe.get("format", {})
    format_name = str(fmt.get("format_name", ""))
    tokens = set(format_name.split(","))
    if "matroska" in tokens:
        return "mkv", format_name
    if "avi" in tokens:
        return "avi", format_name
    if "mp4" in tokens or "mov" in tokens:
        brand = str(fmt.get("tags", {}).get("major_brand", "")).strip().lower()
        return ("mov" if brand == "qt" else "mp4"), format_name
    return None, format_name


def validate_media(probe: dict[str, Any], allowed: list[str], max_duration_s: float,
                   size_bytes: int) -> dict[str, Any]:
    """Check container, audio stream and duration (§3 stage 2). Returns the `media` record."""
    container, format_name = detect_container(probe)
    if container is None or container not in allowed:
        detected = container or format_name or "unknown"
        raise MediaValidationError(
            415, "unsupported_media",
            f"Container '{detected}' is not supported. Allowed: {', '.join(allowed)}.",
            {"detected": detected},
        )
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise MediaValidationError(422, "no_audio_stream", "The file has no audio stream.")
    try:
        duration = float(probe.get("format", {}).get("duration") or audio[0].get("duration"))
    except (TypeError, ValueError):
        raise MediaValidationError(
            422, "unreadable_media", "Could not determine the duration."
        ) from None
    if duration > max_duration_s:
        raise MediaValidationError(
            422, "duration_exceeded",
            f"Duration {duration:.0f}s exceeds the limit of {max_duration_s:.0f}s.",
            {"duration_s": round(duration, 1), "max_duration_s": max_duration_s},
        )
    return {
        "container": format_name,
        "ext": container,
        "duration_s": round(duration, 3),
        "size_bytes": size_bytes,
        "audio_codec": audio[0].get("codec_name"),
    }
