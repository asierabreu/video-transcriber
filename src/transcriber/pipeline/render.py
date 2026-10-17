"""Stage 9: write transcript.json, .txt, .srt and .vtt (§8)."""

from __future__ import annotations

import json
from typing import Any

from transcriber.formats.json_v1 import build_transcript_json
from transcriber.formats.srt import render_srt
from transcriber.formats.txt import render_txt
from transcriber.formats.vtt import render_vtt
from transcriber.pipeline.stages import JobContext
from transcriber.store.files import JobFiles, atomic_write_text

FORMATS = ["json", "txt", "srt", "vtt"]
DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-community-1"


def run(ctx: JobContext) -> None:
    """Render all output formats; transcript.json is written last as the completion marker."""
    cleaned = ctx.files.read_json("cleaned.json")
    summary = ctx.files.read_json("summary.json")
    asr_model = ctx.files.read_json("asr.json").get("model")
    llm_model = cleaned.get("llm_model") or summary.get("llm_model")
    models = {"asr": asr_model, "diarization": DIARIZATION_MODEL_NAME, "llm": llm_model}
    utterances = cleaned["utterances"]

    atomic_write_text(ctx.files.output("txt"), render_txt(utterances))
    atomic_write_text(ctx.files.output("srt"), render_srt(utterances))
    atomic_write_text(ctx.files.output("vtt"), render_vtt(utterances))
    doc = build_transcript_json(ctx.job_id, cleaned, models, list(ctx.warnings))
    atomic_write_text(ctx.files.output("json"), json.dumps(doc, ensure_ascii=False))
    ctx.progress(1.0)


def build_result(files: JobFiles) -> dict[str, Any]:
    """The `result` object of a succeeded job (§7.1), derived from the rendered files."""
    doc = json.loads(files.output("json").read_text(encoding="utf-8"))
    summary = files.read_json("summary.json")
    cleaned_fallbacks = 0
    if files.exists("cleaned.json"):
        cleaned_fallbacks = int(files.read_json("cleaned.json").get("llm_fallbacks", 0))
    models = doc["models"]
    return {
        "speakers": len(doc["speakers"]),
        "words": sum(len(u["words"]) for u in doc["utterances"]),
        "cleanup_applied": doc["cleanup_applied"],
        "summary_available": bool(summary.get("available")),
        "formats": FORMATS,
        "models": {"asr": models.get("asr"), "diarization": "community-1",
                   "llm": models.get("llm")},
        "llm_fallbacks": cleaned_fallbacks,
    }
