"""Stage 6: assign each word to a speaker and group words into utterances (§3)."""

from __future__ import annotations

from typing import Any

from transcriber.pipeline.stages import JobContext

NEAREST_TURN_S = 0.5
MAX_GAP_S = 1.5
MAX_UTTERANCE_S = 60.0


def assign_speakers(words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[str | None]:
    """Speaker label per word.

    A word goes to the turn it overlaps most. With no overlap it goes to the nearest turn within
    0.5 s, otherwise to the previous word's speaker. Both inputs must be sorted by start time.
    """
    labels: list[str | None] = []
    j = 0
    previous: str | None = None
    for word in words:
        ws, we = float(word["start"]), float(word["end"])
        while j < len(turns) and float(turns[j]["end"]) < ws - NEAREST_TURN_S:
            j += 1
        best_label, best_overlap = None, 0.0
        near_label, near_dist = None, NEAREST_TURN_S
        k = j
        while k < len(turns) and float(turns[k]["start"]) <= we + NEAREST_TURN_S:
            ts, te = float(turns[k]["start"]), float(turns[k]["end"])
            overlap = min(we, te) - max(ws, ts)
            if overlap > best_overlap:
                best_label, best_overlap = turns[k]["speaker"], overlap
            dist = max(ts - we, ws - te, 0.0)
            if dist <= near_dist:
                near_label, near_dist = turns[k]["speaker"], dist
            k += 1
        label = best_label or near_label or previous
        labels.append(label)
        if label is not None:
            previous = label
    # Words before the first assignable turn take the first speaker found.
    first = next((lab for lab in labels if lab is not None), None)
    return [lab if lab is not None else first for lab in labels]


def relabel(labels: list[str | None]) -> list[str]:
    """Rename speakers SPEAKER_00, SPEAKER_01, ... in order of first appearance."""
    mapping: dict[str | None, str] = {}
    out = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = f"SPEAKER_{len(mapping):02d}"
        out.append(mapping[lab])
    return out


def group_utterances(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive same-speaker words; split on gaps > 1.5 s or length > 60 s."""
    utterances: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            utterances.append({
                "id": f"u_{len(utterances):04d}",
                "speaker": current[0]["speaker"],
                "start": current[0]["start"],
                "end": max(w["end"] for w in current),
                "raw_text": " ".join(w["w"] for w in current),
                "words": list(current),
            })

    for word in words:
        if current:
            same = word["speaker"] == current[0]["speaker"]
            gap = word["start"] - current[-1]["end"]
            length = word["end"] - current[0]["start"]
            if not same or gap > MAX_GAP_S or length > MAX_UTTERANCE_S:
                flush()
                current = []
        current.append(word)
    flush()
    return utterances


def align(asr: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the aligned artifact from ASR output and diarization turns."""
    words = sorted(asr["words"], key=lambda w: (w["start"], w["end"]))
    turns = sorted(turns, key=lambda t: t["start"])
    labels = relabel(assign_speakers(words, turns)) if turns else ["SPEAKER_00"] * len(words)
    labelled = [{**w, "speaker": lab} for w, lab in zip(words, labels, strict=True)]
    utterances = group_utterances(labelled)
    return {
        "language": asr.get("language", "en"),
        "duration_s": asr.get("duration_s"),
        "speakers": sorted({u["speaker"] for u in utterances}),
        "utterances": utterances,
    }


def run(ctx: JobContext) -> None:
    """Write aligned.json."""
    asr = ctx.files.read_json("asr.json")
    turns = ctx.files.read_json("diarization.json")["turns"] \
        if ctx.files.exists("diarization.json") else []
    ctx.files.write_json("aligned.json", align(asr, turns))
    ctx.progress(1.0)
