"""Job identifiers (ULIDs)."""

from __future__ import annotations

import re

from ulid import ULID

_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def new_job_id() -> str:
    """Return a new, time-sortable job ID."""
    return str(ULID())


def is_valid_job_id(value: str) -> bool:
    """Return True if `value` is a well-formed ULID. Guards every path built from an ID."""
    return bool(_ULID_RE.match(value))
