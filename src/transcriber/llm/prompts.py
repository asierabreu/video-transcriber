"""Prompt templates and JSON schemas for LLM post-processing."""

from __future__ import annotations

import json
from typing import Any

CLEANUP_SYSTEM = (
    "You correct automatic speech-recognition transcripts of English speech.\n"
    "For each utterance you receive, fix punctuation, capitalization and obvious "
    "mis-recognitions (wrong homophones, split or merged words, misheard common terms). "
    "Rules:\n"
    "- Keep the speaker's words and meaning. Do not summarize, paraphrase, translate, add "
    "content or remove content.\n"
    "- Keep filler words only when they carry meaning.\n"
    "- Never merge, split, reorder or drop utterances.\n"
    "- Items with \"context_only\": true are for context only; do not return them.\n"
    "- Return JSON: {\"utterances\": [{\"id\": <same id>, \"text\": <corrected text>}]} with "
    "exactly one entry for every utterance that is not context_only."
)

CLEANUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["utterances"],
}

SUMMARY_MAP_SYSTEM = (
    "You summarize part of a meeting or video transcript. Speakers are labelled "
    "SPEAKER_00, SPEAKER_01, and so on. Write at most 200 words as Markdown bullet points. "
    "Capture the main points, any decisions, and any action items with the speaker who owns "
    "them. Only state what the transcript supports. Output only the bullet points."
)

SUMMARY_REDUCE_SYSTEM = (
    "You merge partial summaries of one transcript into a shorter combined set of Markdown "
    "bullet points (at most 250 words). Keep decisions, action items, owners and timestamps. "
    "Remove repetition. Only state what the partial summaries support. "
    "Output only the bullet points."
)

SUMMARY_FINAL_SYSTEM = (
    "You write the final summary of a transcript from partial summaries. Output Markdown with "
    "exactly these sections, in this order, and nothing else:\n"
    "## Overview\n3-5 sentences.\n"
    "## Key points\nBullet points.\n"
    "## Decisions\nBullet points, or 'None recorded.'\n"
    "## Action items\nBullet points with the owner's speaker label when stated, or "
    "'None recorded.'\n"
    "## Topics\nBullet points of the form '[HH:MM:SS] topic', using the section start times "
    "given in the input.\n"
    "Only state what the partial summaries support."
)


def cleanup_user_message(items: list[dict[str, Any]]) -> str:
    """User message for a cleanup chunk."""
    return "Correct these utterances:\n" + json.dumps({"utterances": items}, ensure_ascii=False)


def summary_map_user_message(start_label: str, transcript: str) -> str:
    """User message for one map step."""
    return f"Transcript section starting at [{start_label}]:\n\n{transcript}"


def summary_reduce_user_message(parts: list[str]) -> str:
    """User message for a reduce step."""
    return "Partial summaries:\n\n" + "\n\n".join(parts)
