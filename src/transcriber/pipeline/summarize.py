"""Stage 8: map-reduce summary with a fixed Markdown structure (§6.4)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any

from transcriber.backends.base import LlmBackend
from transcriber.errors import LLMError
from transcriber.formats import hms
from transcriber.llm.chunking import pack
from transcriber.llm.prompts import (
    SUMMARY_FINAL_SYSTEM,
    SUMMARY_MAP_SYSTEM,
    SUMMARY_REDUCE_SYSTEM,
    summary_map_user_message,
    summary_reduce_user_message,
)
from transcriber.llm.tokens import estimate_tokens, summary_input_budget
from transcriber.pipeline.stages import JobContext

log = logging.getLogger(__name__)

MAX_REDUCE_ROUNDS = 6


def _chat(llm: LlmBackend, system: str, user: str) -> str:
    """One call with a single retry."""
    try:
        return llm.chat(system, user)
    except LLMError:
        return llm.chat(system, user)


def summarize_utterances(
    utterances: list[dict[str, Any]], llm: LlmBackend,
    progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    max_concurrency: int = 1,
) -> tuple[str, dict[str, int]]:
    """Return (summary markdown, counters). Raises LLMError if the LLM keeps failing."""
    budget = summary_input_budget(llm.num_ctx)
    lines = [(u["start"], f"[{hms(u['start'])}] {u['speaker']}: {u['text']}")
             for u in utterances if u["text"].strip()]
    chunks = pack(lines, budget, lambda item: estimate_tokens(item[1]) + 1)

    def map_chunk(chunk: list[tuple[float, str]]) -> str:
        start_label = hms(chunk[0][0])
        text = "\n".join(line for _, line in chunk)
        bullets = _chat(llm, SUMMARY_MAP_SYSTEM, summary_map_user_message(start_label, text))
        return f"Section starting [{start_label}]:\n{bullets}"

    parts_by_index: dict[int, str] = {}
    workers = min(max(1, max_concurrency), len(chunks))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="summary-map") as pool:
        futures = {pool.submit(map_chunk, chunk): index
                   for index, chunk in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), start=1):
            if check_cancel:
                check_cancel()
            parts_by_index[futures[future]] = future.result()
            if progress:
                progress(done / (len(chunks) + 1))

    parts = [parts_by_index[index] for index in range(len(chunks))]
    if not chunks and progress:
        progress(1.0)

    rounds = 0
    while sum(estimate_tokens(p) for p in parts) > budget and len(parts) > 1 \
            and rounds < MAX_REDUCE_ROUNDS:
        rounds += 1
        groups = pack(parts, budget, estimate_tokens)
        if len(groups) == len(parts):
            # Every part already fills a whole request; merge in pairs to make progress.
            groups = [parts[i:i + 2] for i in range(0, len(parts), 2)]
        reduced = []
        for group in groups:
            if check_cancel:
                check_cancel()
            header = group[0].split("\n", 1)[0]
            merged = _chat(llm, SUMMARY_REDUCE_SYSTEM, summary_reduce_user_message(group))
            reduced.append(f"{header}\n{merged}")
        parts = reduced

    final = _chat(llm, SUMMARY_FINAL_SYSTEM, summary_reduce_user_message(parts))
    if progress:
        progress(1.0)
    return final.strip() + "\n", {"maps": len(chunks), "reduce_rounds": rounds}


def run(ctx: JobContext) -> None:
    """Write summary.md (when available) and the summary.json completion marker."""
    if not ctx.options.get("summary", True):
        ctx.files.write_json("summary.json", {"available": False, "reason": "disabled"})
        return
    cleaned = ctx.files.read_json("cleaned.json")
    if not any(u["text"].strip() for u in cleaned["utterances"]):
        ctx.files.write_json("summary.json", {"available": False, "reason": "no speech"})
        return

    llm = ctx.factory.llm()
    status = llm.status()
    if not status.usable:
        if not status.reachable and not ctx.last_attempt:
            raise LLMError(f"LLM runtime unreachable: {status.detail}")
        ctx.warn(f"summary skipped: {status.detail or 'LLM unavailable'}")
        ctx.files.write_json("summary.json", {"available": False, "reason": "llm unavailable"})
        return

    try:
        markdown, stats = summarize_utterances(
            cleaned["utterances"], llm, ctx.progress, ctx.check_cancel,
            ctx.settings.llm_map_concurrency,
        )
    except LLMError as exc:
        if not ctx.last_attempt:
            raise
        ctx.warn(f"summary skipped: {exc}")
        ctx.files.write_json("summary.json", {"available": False, "reason": "llm error"})
        return
    ctx.files.write_text("summary.md", markdown)
    ctx.files.write_json("summary.json", {"available": True, "llm_model": llm.name, **stats})
