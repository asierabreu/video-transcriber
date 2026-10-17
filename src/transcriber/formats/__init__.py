"""Transcript output formats (§8.3–8.4)."""

from __future__ import annotations


def hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def timestamp(seconds: float, decimal_sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (SRT uses ',', VTT uses '.')."""
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_sep}{ms:03d}"
