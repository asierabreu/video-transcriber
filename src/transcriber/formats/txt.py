"""Plain-text transcript output."""

from __future__ import annotations

from typing import Any

from transcriber.formats import hms


def render_txt(utterances: list[dict[str, Any]]) -> str:
    """One paragraph per speaker turn: `[HH:MM:SS] SPEAKER_00: text`.

    Consecutive utterances by the same speaker are merged into one paragraph.
    """
    paragraphs: list[tuple[float, str, list[str]]] = []
    for utt in utterances:
        text = utt["text"].strip()
        if not text:
            continue
        if paragraphs and paragraphs[-1][1] == utt["speaker"]:
            paragraphs[-1][2].append(text)
        else:
            paragraphs.append((utt["start"], utt["speaker"], [text]))
    lines = [f"[{hms(start)}] {speaker}: {' '.join(texts)}"
             for start, speaker, texts in paragraphs]
    return "\n\n".join(lines) + ("\n" if lines else "")
