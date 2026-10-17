from transcriber.formats import hms, timestamp
from transcriber.formats.cues import MAX_CUE_CHARS, MAX_CUE_SECONDS, build_cues, wrap_lines
from transcriber.formats.json_v1 import build_transcript_json
from transcriber.formats.srt import render_srt
from transcriber.formats.txt import render_txt
from transcriber.formats.vtt import render_vtt


def utt(uid, speaker, start, words, text=None):
    ws = [{"w": t, "start": start + i * 0.5, "end": start + i * 0.5 + 0.4, "p": 0.8,
           "speaker": speaker} for i, t in enumerate(words)]
    raw = " ".join(words)
    return {"id": uid, "speaker": speaker, "start": start, "end": ws[-1]["end"],
            "raw_text": raw, "text": text or raw, "words": ws}


def test_timestamps():
    assert hms(3725.9) == "01:02:05"
    assert timestamp(3725.5, ",") == "01:02:05,500"
    assert timestamp(0.0004, ".") == "00:00:00.000"


def test_txt_merges_consecutive_same_speaker():
    utts = [utt("u_0000", "SPEAKER_00", 0, ["hello"]), utt("u_0001", "SPEAKER_00", 3, ["again"]),
            utt("u_0002", "SPEAKER_01", 6, ["hi"])]
    assert render_txt(utts) == ("[00:00:00] SPEAKER_00: hello again\n\n"
                                "[00:00:06] SPEAKER_01: hi\n")


def test_cues_respect_limits():
    words = [f"word{i:02d}" for i in range(40)]
    cues = build_cues([utt("u_0000", "SPEAKER_00", 0, words)])
    assert len(cues) > 1
    for cue in cues:
        assert cue.end - cue.start <= MAX_CUE_SECONDS + 0.5
        assert len(" ".join(cue.lines)) <= MAX_CUE_CHARS
        assert len(cue.lines) <= 2
    for a, b in zip(cues, cues[1:], strict=False):
        assert a.end <= b.start


def test_min_duration_and_no_overlap():
    u1 = utt("u_0000", "SPEAKER_00", 0.0, ["a"])
    u2 = utt("u_0001", "SPEAKER_01", 0.5, ["b"])
    cues = build_cues([u1, u2])
    assert cues[0].end <= cues[1].start
    assert cues[1].end - cues[1].start >= 0.8


def test_cleaned_text_with_different_token_count_keeps_timings_inside_utterance():
    u = utt("u_0000", "SPEAKER_00", 10.0, ["road", "map", "is", "ready"], text="Roadmap is ready.")
    cue = build_cues([u])[0]
    assert cue.start >= 10.0 and cue.end <= u["end"] + 0.8
    assert cue.lines == ["Roadmap is ready."]


def test_wrap_lines_balances_two_lines():
    lines = wrap_lines("the quick brown fox jumps over the lazy dog and keeps running far")
    assert len(lines) == 2 and all(len(line) <= 42 for line in lines)


def test_srt_and_vtt_shapes():
    utts = [utt("u_0000", "SPEAKER_00", 1.0, ["hello", "<world>"])]
    srt = render_srt(utts)
    assert srt.startswith("1\n00:00:01,000 --> ")
    assert "[SPEAKER_00] hello <world>" in srt
    vtt = render_vtt(utts)
    assert vtt.startswith("WEBVTT\n\n00:00:01.000 --> ")
    assert "<v SPEAKER_00>hello &lt;world&gt;" in vtt
    assert render_srt([]) == "" and render_vtt([]).strip() == "WEBVTT"


def test_transcript_json_schema():
    cleaned = {"language": "en", "duration_s": 9.0, "cleanup_applied": True,
               "utterances": [utt("u_0000", "SPEAKER_00", 0, ["hi", "there"], "Hi there.")]}
    doc = build_transcript_json("01ABC", cleaned, {"asr": "m", "diarization": "d", "llm": "l"},
                                ["w1"])
    assert doc["schema_version"] == 1
    assert doc["speakers"] == [{"id": "SPEAKER_00", "talk_time_s": 0.9, "words": 2}]
    u = doc["utterances"][0]
    assert u["text"] == "Hi there." and u["raw_text"] == "hi there"
    assert u["confidence"] == 0.8
    assert doc["warnings"] == ["w1"]
