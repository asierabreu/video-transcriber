"""Shared fixtures: settings, API client, fake backends and generated media."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transcriber.api.main import create_app
from transcriber.backends.base import AsrResult, LlmStatus, SpeakerTurn
from transcriber.config import Settings
from transcriber.errors import LLMError
from transcriber.hardware.tiers import AsrAttempt, HardwareProfile, resolve_effective
from transcriber.store.db import open_db
from transcriber.worker.runner import JobRunner

API_KEY = "test-key-0123456789abcdef0123456789abcdef"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        api_key=API_KEY, data_dir=tmp_path / "data", models_dir=tmp_path / "models",
        log_format="console", min_free_disk_gb=0, retention_days=2,
        ollama_url="http://127.0.0.1:9",  # nothing listens here
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings), headers={"X-API-Key": API_KEY})


# ---------- generated media ----------

def make_media(path: Path, *, seconds: float = 3.0, audio: bool = True,
               video: bool = True) -> Path:
    """Generate a small test file with ffmpeg's lavfi sources; container from the suffix."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if video:
        cmd += ["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}"]
    if path.suffix == ".avi":
        cmd += ["-c:v", "mpeg4"] if video else []
        cmd += ["-c:a", "pcm_s16le"] if audio else []
    elif video:
        cmd += ["-c:v", "mpeg4"]
    if audio and path.suffix in (".mp4", ".mov", ".mkv"):
        cmd += ["-c:a", "aac"]
    cmd += ["-shortest", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    d = tmp_path / "media"
    d.mkdir()
    return d


# ---------- fake backends ----------

class FakeAsr:
    def __init__(self, words: list[dict[str, Any]] | None = None) -> None:
        self.name = "fake-asr"
        self.words = words
        self.closed = False

    def transcribe(self, audio_path: Path, progress: Any) -> AsrResult:
        assert audio_path.exists()
        words = self.words if self.words is not None else [
            {"w": w, "start": i * 0.4, "end": i * 0.4 + 0.3, "p": 0.9}
            for i, w in enumerate(
                ["hello", "there", "this", "is", "a", "test", "of", "the", "system", "ok"]
            )
        ]
        progress(1.0)
        return {"language": "en", "duration_s": 5.0, "words": list(words),
                "segments": [{"start": 0.0, "end": 5.0, "text": "", "n_words": len(words)}]}

    def close(self) -> None:
        self.closed = True


class FakeDiarizer:
    def __init__(self, turns: list[SpeakerTurn] | None = None) -> None:
        self.name = "fake-diarizer"
        self.turns = turns if turns is not None else [
            {"start": 0.0, "end": 1.9, "speaker": "SPK_A"},
            {"start": 2.0, "end": 5.0, "speaker": "SPK_B"},
        ]

    def diarize(self, audio_path: Path, num_speakers: Any, min_speakers: Any,
                max_speakers: Any) -> list[SpeakerTurn]:
        return list(self.turns)

    def close(self) -> None:
        pass


class FakeLlm:
    """Cleanup capitalizes and adds a period; summaries return fixed Markdown."""

    def __init__(self, num_ctx: int = 8192, reachable: bool = True, present: bool = True,
                 mode: str = "good") -> None:
        self.name = "fake-llm"
        self.num_ctx = num_ctx
        self.reachable = reachable
        self.present = present
        self.mode = mode
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def status(self) -> LlmStatus:
        return LlmStatus(self.reachable, self.present, "" if self.present else "not pulled")

    def chat(self, system: str, user: str, *, json_schema: Any = None,
             max_tokens: Any = None) -> str:
        with self.lock:
            self.calls.append((system, user))
        if self.mode == "down":
            raise LLMError("connection refused")
        if json_schema is not None:
            if self.mode == "garbage":
                return "not json"
            payload = json.loads(user.split("\n", 1)[1])
            items = [i for i in payload["utterances"] if not i.get("context_only")]
            if self.mode == "rewrite":
                return json.dumps({"utterances": [
                    {"id": i["id"], "text": "completely different words here now"}
                    for i in items]})
            if self.mode == "drop":
                items = items[1:]
            return json.dumps({"utterances": [
                {"id": i["id"], "text": i["text"].capitalize() + "."} for i in items]})
        if "final summary" in system:
            return ("## Overview\nA test.\n## Key points\n- Point\n## Decisions\nNone recorded.\n"
                    "## Action items\nNone recorded.\n## Topics\n- [00:00:00] Testing")
        return "- a summarized point about the section"


class FakeFactory:
    def __init__(self, asr: FakeAsr | None = None, diarizer: FakeDiarizer | None = None,
                 llm: FakeLlm | None = None, asr_errors: list[Exception] | None = None,
                 diar_errors: list[Exception] | None = None,
                 models: set[str] | None = None) -> None:
        self._asr = asr or FakeAsr()
        self._diar = diarizer or FakeDiarizer()
        self._llm = llm or FakeLlm()
        self.asr_errors = list(asr_errors or [])
        self.diar_errors = list(diar_errors or [])
        self.models = models if models is not None else {"distil-large-v3"}
        self.asr_attempts: list[AsrAttempt] = []
        self.diar_devices: list[str] = []

    def available_asr_models(self) -> set[str]:
        return self.models

    def asr(self, attempt: AsrAttempt) -> FakeAsr:
        self.asr_attempts.append(attempt)
        if self.asr_errors:
            raise self.asr_errors.pop(0)
        return self._asr

    def diarizer(self, device: str) -> FakeDiarizer:
        self.diar_devices.append(device)
        if self.diar_errors:
            raise self.diar_errors.pop(0)
        return self._diar

    def llm(self) -> FakeLlm:
        return self._llm


def make_runner(settings: Settings, factory: FakeFactory, device: str = "cpu",
                stop_event: threading.Event | None = None) -> tuple[JobRunner, Any]:
    profile = HardwareProfile(device=device, tier="cpu" if device == "cpu" else "cuda-large",
                              cpu_threads=2)
    eff = resolve_effective(settings, profile)
    conn = open_db(settings.db_path)
    runner = JobRunner(settings, conn, profile, eff, factory, stop_event or threading.Event())
    return runner, conn
