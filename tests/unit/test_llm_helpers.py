from transcriber.llm.chunking import pack
from transcriber.llm.guard import accept_edit, edit_ratio, normalize_words
from transcriber.llm.tokens import cleanup_input_budget, estimate_tokens, summary_input_budget


def test_token_estimate_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefg") == 2


def test_budgets_for_8k_context():
    assert 3000 <= cleanup_input_budget(8192) <= 3600
    assert 5500 <= summary_input_budget(8192) <= 6500
    assert cleanup_input_budget(16384) > cleanup_input_budget(8192)


def test_pack_respects_budget_and_never_splits():
    chunks = pack([3, 3, 3, 10, 1], budget=6, cost=lambda x: x)
    assert chunks == [[3, 3], [3], [10], [1]]
    assert pack([], 5, lambda x: x) == []


def test_normalize_ignores_case_and_punctuation():
    assert normalize_words("Let's GO, now!") == ["lets", "go", "now"]


def test_edit_ratio_and_guard():
    original = "good morning everyone lets start with the road map"
    cleaned = "Good morning, everyone. Let's start with the roadmap."
    assert edit_ratio(original, cleaned) == 2 / 9
    assert accept_edit(original, cleaned, 0.25)
    assert not accept_edit(original, "Something else entirely was said here.", 0.25)
    assert not accept_edit("some words", "", 0.25)
    assert accept_edit("", "", 0.25)
