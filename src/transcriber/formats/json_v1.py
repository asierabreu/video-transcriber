"""`transcript.json` schema v1 (§8.3)."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1


def speaker_stats(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Talk time and word count per speaker, ordered by speaker label."""
    stats: dict[str, dict[str, Any]] = {}
    for utt in utterances:
        s = stats.setdefault(utt["speaker"], {"id": utt["speaker"], "talk_time_s": 0.0,
                                               "words": 0})
        s["talk_time_s"] += max(0.0, utt["end"] - utt["start"])
        s["words"] += len(utt.get("words") or [])
    for s in stats.values():
        s["talk_time_s"] = round(s["talk_time_s"], 1)
    return [stats[k] for k in sorted(stats)]


def build_transcript_json(
    job_id: str, cleaned: dict[str, Any], models: dict[str, str | None],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the public transcript document from the cleaned stage artifact."""
    utterances = []
    for utt in cleaned["utterances"]:
        words = utt.get("words") or []
        probs = [w["p"] for w in words if w.get("p") is not None]
        utterances.append({
            "id": utt["id"],
            "speaker": utt["speaker"],
            "start": round(utt["start"], 3),
            "end": round(utt["end"], 3),
            "text": utt["text"],
            "raw_text": utt["raw_text"],
            "confidence": round(sum(probs) / len(probs), 3) if probs else None,
            "words": words,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "language": cleaned.get("language", "en"),
        "duration_s": cleaned.get("duration_s"),
        "models": models,
        "cleanup_applied": bool(cleaned.get("cleanup_applied")),
        "speakers": speaker_stats(cleaned["utterances"]),
        "utterances": utterances,
        "warnings": warnings,
    }
