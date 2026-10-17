"""SubRip (.srt) output."""

from __future__ import annotations

from typing import Any

from transcriber.formats import timestamp
from transcriber.formats.cues import build_cues


def render_srt(utterances: list[dict[str, Any]]) -> str:
    """Render utterances as SRT, prefixing each cue with `[SPEAKER_xx] `."""
    prefix_len = len("[SPEAKER_00] ")
    blocks = []
    for index, cue in enumerate(build_cues(utterances, prefix_len), start=1):
        lines = list(cue.lines)
        lines[0] = f"[{cue.speaker}] {lines[0]}"
        blocks.append(
            f"{index}\n{timestamp(cue.start, ',')} --> {timestamp(cue.end, ',')}\n"
            + "\n".join(lines)
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
