"""Stage 7: LLM cleanup of utterance text with chunking and the edit guard (§6.4)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from transcriber.backends.base import LlmBackend
from transcriber.errors import LLMError
from transcriber.llm.chunking import pack
from transcriber.llm.guard import accept_edit
from transcriber.llm.prompts import CLEANUP_SCHEMA, CLEANUP_SYSTEM, cleanup_user_message
from transcriber.llm.tokens import cleanup_input_budget, cleanup_output_tokens, estimate_tokens
from transcriber.pipeline.stages import JobContext

log = logging.getLogger(__name__)

CONTEXT_UTTERANCES = 2


class _ChunkFailed(Exception):
    def __init__(self, message: str, transport: bool) -> None:
        super().__init__(message)
        self.transport = transport


def _item(utt: dict[str, Any], context_only: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {"id": utt["id"], "speaker": utt["speaker"], "text": utt["raw_text"]}
    if context_only:
        item["context_only"] = True
    return item


def _parse_reply(reply: str) -> dict[str, str]:
    data = json.loads(reply)
    entries = data["utterances"] if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) \
                and isinstance(entry.get("text"), str):
            out[entry["id"]] = entry["text"].strip()
    return out


def _clean_chunk(llm: LlmBackend, items: list[dict[str, Any]], input_tokens: int
                 ) -> dict[str, str]:
    """Send one chunk, retrying once on transport or parse failure."""
    last: _ChunkFailed | None = None
    for _ in range(2):
        try:
            reply = llm.chat(CLEANUP_SYSTEM, cleanup_user_message(items),
                             json_schema=CLEANUP_SCHEMA,
                             max_tokens=cleanup_output_tokens(input_tokens))
        except LLMError as exc:
            last = _ChunkFailed(str(exc), transport=True)
            continue
        try:
            return _parse_reply(reply)
        except (ValueError, KeyError, TypeError) as exc:
            last = _ChunkFailed(f"invalid JSON from LLM: {exc}", transport=False)
    assert last is not None
    raise last


def clean_utterances(
    utterances: list[dict[str, Any]], llm: LlmBackend, max_edit_ratio: float,
    progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return utterances with a `text` field, plus counters.

    Counters: `fallbacks` (utterances that kept their original text), `extra_ids` (IDs the
    model invented), `failed_chunks`, `transport_failures`, `chunks`.
    """
    budget = cleanup_input_budget(llm.num_ctx)
    index_of = {u["id"]: i for i, u in enumerate(utterances)}

    def cost(u: dict[str, Any]) -> int:
        return estimate_tokens(json.dumps(_item(u), ensure_ascii=False))

    chunks = pack(utterances, budget, cost)
    out = [{**u, "text": u["raw_text"]} for u in utterances]
    stats = {"fallbacks": 0, "extra_ids": 0, "failed_chunks": 0, "transport_failures": 0,
             "chunks": len(chunks)}
    for n, chunk in enumerate(chunks):
        if check_cancel:
            check_cancel()
        first = index_of[chunk[0]["id"]]
        context = [_item(u, context_only=True)
                   for u in utterances[max(0, first - CONTEXT_UTTERANCES):first]]
        items = context + [_item(u) for u in chunk]
        wanted = {u["id"] for u in chunk}
        try:
            cleaned = _clean_chunk(llm, items, sum(cost(u) for u in chunk))
        except _ChunkFailed as exc:
            log.warning("cleanup chunk failed", extra={"chunk": n, "error": str(exc)})
            stats["failed_chunks"] += 1
            stats["transport_failures"] += int(exc.transport)
            cleaned = {}
        else:
            stats["extra_ids"] += len(set(cleaned) - wanted - {c["id"] for c in context})
        for utt in chunk:
            new_text = cleaned.get(utt["id"])
            if new_text is not None and accept_edit(utt["raw_text"], new_text, max_edit_ratio):
                out[index_of[utt["id"]]]["text"] = new_text
            else:
                # Missing ID, failed chunk or rejected by the guard: keep the original text.
                stats["fallbacks"] += 1
        if progress:
            progress((n + 1) / len(chunks))
    return out, stats


def run(ctx: JobContext) -> None:
    """Write cleaned.json (a copy of aligned.json with `text` per utterance)."""
    aligned = ctx.files.read_json("aligned.json")
    utterances = aligned["utterances"]
    base = {**aligned, "cleanup_applied": False, "llm_fallbacks": 0, "llm_model": None}
    if not ctx.options.get("cleanup", True) or not utterances:
        base["utterances"] = [{**u, "text": u["raw_text"]} for u in utterances]
        ctx.files.write_json("cleaned.json", base)
        return

    llm = ctx.factory.llm()
    status = llm.status()
    if not status.usable:
        if not status.reachable and not ctx.last_attempt:
            raise LLMError(f"LLM runtime unreachable: {status.detail}")
        ctx.warn(f"cleanup skipped: {status.detail or 'LLM unavailable'}")
        base["utterances"] = [{**u, "text": u["raw_text"]} for u in utterances]
        base["llm_fallbacks"] = len(utterances)
        ctx.files.write_json("cleaned.json", base)
        return

    cleaned, stats = clean_utterances(utterances, llm, ctx.settings.llm_max_edit_ratio,
                                      ctx.progress, ctx.check_cancel)
    if stats["chunks"] and stats["transport_failures"] == stats["chunks"] \
            and not ctx.last_attempt:
        raise LLMError("every cleanup request failed to reach the LLM")
    if stats["fallbacks"]:
        ctx.warn(f"cleanup: kept the original text for {stats['fallbacks']} of "
                 f"{len(utterances)} utterances (edit guard or LLM errors)")
    if stats["extra_ids"]:
        ctx.warn(f"cleanup: ignored {stats['extra_ids']} utterance IDs invented by the LLM")
    ctx.files.write_json("cleaned.json", {
        **aligned, "utterances": cleaned, "cleanup_applied": True,
        "llm_fallbacks": stats["fallbacks"], "llm_model": llm.name,
    })
