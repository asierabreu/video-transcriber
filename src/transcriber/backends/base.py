"""Backend interfaces. Pipeline stages depend only on these protocols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

ProgressFn = Callable[[float], None]


class AsrWord(TypedDict):
    """One recognized word. `w` has no surrounding whitespace."""

    w: str
    start: float
    end: float
    p: float


class AsrResult(TypedDict):
    """Output of an ASR backend (stored as asr.json)."""

    language: str
    duration_s: float
    words: list[AsrWord]
    segments: list[dict[str, Any]]


class SpeakerTurn(TypedDict):
    """One diarization turn (stored in diarization.json)."""

    start: float
    end: float
    speaker: str


class AsrBackend(Protocol):
    """Speech recognition."""

    name: str

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> AsrResult:
        """Transcribe a 16 kHz mono WAV. `progress` receives a fraction in [0, 1]."""
        ...

    def close(self) -> None:
        """Release model memory (GPU included)."""
        ...


class DiarizationBackend(Protocol):
    """Speaker diarization."""

    name: str

    def diarize(
        self, audio_path: Path, num_speakers: int | None, min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerTurn]:
        """Return non-overlapping (exclusive) speaker turns sorted by start time."""
        ...

    def close(self) -> None:
        """Release model memory (GPU included)."""
        ...


@dataclass
class LlmStatus:
    """Reachability of the LLM runtime and the configured model."""

    reachable: bool
    model_present: bool
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.reachable and self.model_present


class LlmBackend(Protocol):
    """Chat-style LLM used for cleanup and summaries."""

    name: str
    num_ctx: int

    def status(self) -> LlmStatus:
        """Check that the runtime is reachable and the model is installed."""
        ...

    def chat(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's reply text. Raises LLMError on transport failures."""
        ...
