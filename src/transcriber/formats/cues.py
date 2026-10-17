"""Subtitle cue building shared by SRT and VTT (§8.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_CUE_SECONDS = 7.0
MAX_CUE_CHARS = 84
LINE_CHARS = 42
MIN_CUE_SECONDS = 0.8


@dataclass
class Cue:
    """One subtitle cue."""

    start: float
    end: float
    speaker: str
    lines: list[str]


def _token_timings(utt: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Pair each display token with a time span.

    Display tokens come from the (possibly LLM-cleaned) text; timings come from the raw ASR
    words. When counts differ, tokens are mapped to words proportionally.
    """
    tokens = utt["text"].split()
    words = utt.get("words") or []
    if not tokens:
        return []
    if not words:
        span = (utt["end"] - utt["start"]) / len(tokens)
        return [(t, utt["start"] + i * span, utt["start"] + (i + 1) * span)
                for i, t in enumerate(tokens)]
    n, m = len(tokens), len(words)
    out = []
    for i, tok in enumerate(tokens):
        w = words[min(m - 1, i * m // n)]
        out.append((tok, float(w["start"]), float(w["end"])))
    return out


def wrap_lines(text: str, width: int = LINE_CHARS) -> list[str]:
    """Greedy word wrap; prefers two balanced lines when the text needs wrapping."""
    if len(text) <= width:
        return [text]
    words = text.split()
    best: list[str] | None = None
    best_score = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if len(a) <= width and len(b) <= width:
            score = abs(len(a) - len(b))
            if best_score is None or score < best_score:
                best, best_score = [a, b], score
    if best is not None:
        return best
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_cues(utterances: list[dict[str, Any]], prefix_len: int = 0) -> list[Cue]:
    """Split utterances into cues of at most 7 s and 84 characters, at word boundaries."""
    char_limit = max(20, MAX_CUE_CHARS - prefix_len)
    cues: list[Cue] = []
    for utt in utterances:
        current: list[tuple[str, float, float]] = []
        for tok in _token_timings(utt):
            if current:
                text_len = len(" ".join(t[0] for t in current)) + 1 + len(tok[0])
                duration = tok[2] - current[0][1]
                if text_len > char_limit or duration > MAX_CUE_SECONDS:
                    cues.append(_make_cue(current, utt["speaker"]))
                    current = []
            current.append(tok)
        if current:
            cues.append(_make_cue(current, utt["speaker"]))
    _fix_timing(cues)
    return cues


def _make_cue(tokens: list[tuple[str, float, float]], speaker: str) -> Cue:
    text = " ".join(t[0] for t in tokens)
    start = min(t[1] for t in tokens)
    end = max(t[2] for t in tokens)
    return Cue(start=start, end=end, speaker=speaker, lines=wrap_lines(text))


def _fix_timing(cues: list[Cue]) -> None:
    """Enforce the minimum duration and remove overlaps."""
    cues.sort(key=lambda c: c.start)
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < len(cues) else None
        if cue.end - cue.start < MIN_CUE_SECONDS:
            cue.end = cue.start + MIN_CUE_SECONDS
        if next_start is not None and cue.end > next_start:
            cue.end = max(cue.start + 0.001, next_start)
