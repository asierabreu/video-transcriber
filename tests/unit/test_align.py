from transcriber.pipeline.align import align, assign_speakers, group_utterances, relabel


def w(text, start, end, p=0.9):
    return {"w": text, "start": start, "end": end, "p": p}


def test_assign_by_max_overlap():
    words = [w("a", 0.0, 0.5), w("b", 0.9, 1.4), w("c", 2.0, 2.4)]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "X"}, {"start": 1.0, "end": 3.0, "speaker": "Y"}]
    # "b" overlaps X by 0.1 s and Y by 0.4 s
    assert assign_speakers(words, turns) == ["X", "Y", "Y"]


def test_zero_overlap_uses_nearest_turn_within_half_second():
    words = [w("a", 1.2, 1.4)]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "X"}, {"start": 3.0, "end": 4.0, "speaker": "Y"}]
    assert assign_speakers(words, turns) == ["X"]


def test_far_word_inherits_previous_speaker():
    words = [w("a", 0.1, 0.4), w("b", 5.0, 5.2)]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "X"},
             {"start": 9.0, "end": 10.0, "speaker": "Y"}]
    assert assign_speakers(words, turns) == ["X", "X"]


def test_leading_unassigned_words_take_first_speaker():
    words = [w("a", 0.0, 0.2), w("b", 5.0, 5.2)]
    turns = [{"start": 5.0, "end": 6.0, "speaker": "Y"}]
    assert assign_speakers(words, turns) == ["Y", "Y"]


def test_relabel_in_order_of_appearance():
    assert relabel(["B", "A", "B", None]) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00",
                                              "SPEAKER_02"]


def test_grouping_splits_on_speaker_gap_and_length():
    words = [
        {**w("a", 0.0, 0.5), "speaker": "S0"},
        {**w("b", 0.6, 1.0), "speaker": "S0"},
        {**w("c", 1.1, 1.5), "speaker": "S1"},  # speaker change
        {**w("d", 3.5, 3.9), "speaker": "S1"},  # gap > 1.5 s
    ]
    utts = group_utterances(words)
    assert [u["raw_text"] for u in utts] == ["a b", "c", "d"]
    assert [u["id"] for u in utts] == ["u_0000", "u_0001", "u_0002"]

    long_words = [{**w("x", i * 1.0, i * 1.0 + 0.9), "speaker": "S0"} for i in range(70)]
    assert len(group_utterances(long_words)) == 2


def test_align_without_turns_is_single_speaker():
    asr = {"language": "en", "duration_s": 2.0, "words": [w("hi", 0.0, 0.3), w("you", 0.4, 0.6)]}
    out = align(asr, [])
    assert out["speakers"] == ["SPEAKER_00"]
    assert out["utterances"][0]["raw_text"] == "hi you"


def test_align_empty_transcript():
    out = align({"language": "en", "duration_s": 1.0, "words": []}, [])
    assert out["utterances"] == [] and out["speakers"] == []
