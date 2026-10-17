import json

from tests.conftest import FakeLlm
from transcriber.pipeline.cleanup import clean_utterances
from transcriber.pipeline.summarize import summarize_utterances


def utterances(n, words_each=8):
    out = []
    for i in range(n):
        text = " ".join(f"word{j}" for j in range(words_each))
        out.append({"id": f"u_{i:04d}", "speaker": f"SPEAKER_0{i % 2}", "start": i * 5.0,
                    "end": i * 5.0 + 4.0, "raw_text": text, "words": []})
    return out


def test_cleanup_applies_accepted_edits_and_chunks():
    llm = FakeLlm(num_ctx=2048)
    utts = utterances(60)
    out, stats = clean_utterances(utts, llm, 0.25)
    assert stats["chunks"] > 1 and stats["fallbacks"] == 0
    assert all(u["text"].endswith(".") and u["text"][0].isupper() for u in out)
    assert all(u["raw_text"] == o["raw_text"] for u, o in zip(utts, out, strict=True))
    # Context items are sent but never expected back.
    second_chunk = json.loads(llm.calls[1][1].split("\n", 1)[1])["utterances"]
    assert second_chunk[0]["context_only"] is True


def test_cleanup_guard_rejects_rewrites():
    out, stats = clean_utterances(utterances(3), FakeLlm(mode="rewrite"), 0.25)
    assert stats["fallbacks"] == 3
    assert [u["text"] for u in out] == [u["raw_text"] for u in utterances(3)]


def test_cleanup_invalid_json_keeps_original_after_retry():
    llm = FakeLlm(mode="garbage")
    out, stats = clean_utterances(utterances(2), llm, 0.25)
    assert stats["failed_chunks"] == 1 and stats["fallbacks"] == 2
    assert len(llm.calls) == 2  # one retry
    assert out[0]["text"] == out[0]["raw_text"]


def test_cleanup_missing_ids_keep_original():
    out, stats = clean_utterances(utterances(3), FakeLlm(mode="drop"), 0.25)
    assert stats["fallbacks"] == 1
    assert out[0]["text"] == out[0]["raw_text"] and out[1]["text"].endswith(".")


def test_cleanup_transport_failures_are_counted():
    _, stats = clean_utterances(utterances(2), FakeLlm(mode="down"), 0.25)
    assert stats["transport_failures"] == stats["chunks"] == 1


def test_summary_map_reduce_with_small_context():
    llm = FakeLlm(num_ctx=2048)
    utts = [{**u, "text": u["raw_text"] * 3} for u in utterances(80, words_each=20)]
    md, stats = summarize_utterances(utts, llm)
    assert stats["maps"] > 1
    assert md.startswith("## Overview")
    assert "final summary" in llm.calls[-1][0]
