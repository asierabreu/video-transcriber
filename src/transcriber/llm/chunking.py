"""Greedy packing of whole utterances into token-bounded chunks (§6.4)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def pack(items: Sequence[T], budget: int, cost: Callable[[T], int]) -> list[list[T]]:
    """Split `items` into consecutive chunks whose total cost stays within `budget`.

    Items are never split. An item that alone exceeds the budget becomes its own chunk.
    """
    chunks: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        c = cost(item)
        if current and used + c > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += c
    if current:
        chunks.append(current)
    return chunks


def pack_texts(texts: Sequence[str], budget: int, cost: Callable[[str], int]) -> list[list[str]]:
    """Convenience wrapper of `pack` for plain strings."""
    return pack(list(texts), budget, cost)
