"""WebVTT (.vtt) output."""

from __future__ import annotations

from typing import Any

from transcriber.formats import timestamp
from transcriber.formats.cues import build_cues


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_vtt(utterances: list[dict[str, Any]]) -> str:
    """Render utterances as WebVTT using voice tags (`<v SPEAKER_00>`)."""
    parts = ["WEBVTT", ""]
    for cue in build_cues(utterances):
        lines = [_escape(line) for line in cue.lines]
        lines[0] = f"<v {cue.speaker}>{lines[0]}"
        parts.append(f"{timestamp(cue.start, '.')} --> {timestamp(cue.end, '.')}")
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts)
