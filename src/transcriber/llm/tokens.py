"""Token estimates and context budgets (§6.4)."""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 3.5  # deliberately conservative for English
PROMPT_OVERHEAD_TOKENS = 400
SUMMARY_MAP_OUTPUT_TOKENS = 600
SAFETY_MARGIN = 0.10


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cleanup_input_budget(num_ctx: int) -> int:
    """Input tokens per cleanup request; output is about the same size as input."""
    usable = num_ctx - PROMPT_OVERHEAD_TOKENS - int(num_ctx * SAFETY_MARGIN)
    return max(256, usable // 2)


def summary_input_budget(num_ctx: int) -> int:
    """Input tokens per summary map or reduce request."""
    usable = (num_ctx - PROMPT_OVERHEAD_TOKENS - SUMMARY_MAP_OUTPUT_TOKENS
              - int(num_ctx * SAFETY_MARGIN))
    return max(256, usable)


def cleanup_output_tokens(input_tokens: int) -> int:
    """Generation cap for a cleanup request (JSON overhead adds ~25%)."""
    return int(input_tokens * 1.25) + 128
