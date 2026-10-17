"""Full stack with real models (opt-in: `pytest -m e2e` or `make test-e2e`).

The fixture is built on first run from two public-domain LibriVox recordings by different
readers. Every LibriVox file opens with the same disclaimer, which serves as the reference text.
Clips are cached in tests/fixtures/e2e_cache/. The test skips when the network, models or ML
libraries are unavailable.

Environment overrides:
  E2E_CLIP_URLS   two comma-separated audio URLs (defaults below)
  E2E_ASR_MODEL   faster-whisper model (default base.en)
  E2E_LLM_MODEL   Ollama model (default qwen3:0.6b); LLM steps are skipped if Ollama is down
  MODELS_DIR, OLLAMA_URL as for the service
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import API_KEY
from transcriber.api.main import create_app
from transcriber.config import Settings
from transcriber.llm.guard import normalize_words
from transcriber.store import jobs_repo as repo
from transcriber.store.db import open_db

DEFAULT_CLIPS = [
    "https://archive.org/download/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3",
    "https://archive.org/download/war_worlds_solo_librivox/warofworlds_b1_ch01_wells_64kb.mp3",
]
REFERENCE = "this is a librivox recording all librivox recordings are in the public domain"
CLIP_SECONDS = 25
CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "e2e_cache"


def substring_wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate of `reference` against its best-matching span of `hypothesis`."""
    n = len(reference)
    prev = [0] * (len(hypothesis) + 1)  # free start anywhere in the hypothesis
    for i in range(1, n + 1):
        cur = [i] + [0] * len(hypothesis)
        for j in range(1, len(hypothesis) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (reference[i - 1] != hypothesis[j - 1]))
        prev = cur
    return min(prev) / n


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        pytest.skip(f"cannot download e2e clip {url}: {exc}")
    tmp.replace(dest)


def build_fixture() -> Path:
    """Two 25 s clips by different readers, 1 s apart, muxed into an MP4 with test video."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    CACHE.mkdir(parents=True, exist_ok=True)
    urls = os.environ.get("E2E_CLIP_URLS", ",".join(DEFAULT_CLIPS)).split(",")
    sources = []
    for i, url in enumerate(urls[:2]):
        dest = CACHE / f"clip{i}{Path(url).suffix or '.mp3'}"
        _fetch(url.strip(), dest)
        sources.append(dest)
    out = CACHE / "two_speakers.mp4"
    if out.exists():
        return out
    total = 2 * CLIP_SECONDS + 1
    filt = (f"[0:a]atrim=0:{CLIP_SECONDS},aresample=16000,aformat=channel_layouts=mono[a0];"
            f"[1:a]atrim=0:{CLIP_SECONDS},aresample=16000,aformat=channel_layouts=mono[a1];"
            "anullsrc=r=16000:cl=mono,atrim=0:1[s];[a0][s][a1]concat=n=3:v=0:a=1[a]")
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(sources[0]), "-i", str(sources[1]),
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=5:duration={total}",
        "-filter_complex", filt, "-map", "2:v", "-map", "[a]",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(out),
    ], check=True, capture_output=True)
    return out


def _ensure_asr_model(name: str, models_dir: Path) -> None:
    target = models_dir / "whisper" / name
    if target.is_dir():
        return
    from faster_whisper import download_model

    try:
        download_model(name, output_dir=str(target))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot download ASR model {name}: {exc}")


@pytest.mark.e2e
def test_full_stack_two_speakers(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper")
    pytest.importorskip("pyannote.audio")
    video = build_fixture()

    asr_model = os.environ.get("E2E_ASR_MODEL", "base.en")
    llm_model = os.environ.get("E2E_LLM_MODEL", "qwen3:0.6b")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        api_key=API_KEY, data_dir=tmp_path / "data",
        models_dir=Path(os.environ.get("MODELS_DIR", "./.models")).resolve(),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        asr_model=asr_model, llm_model=llm_model, min_free_disk_gb=0, log_format="console",
    )
    if not (settings.diarization_model_path / "config.yaml").is_file():
        pytest.skip(f"diarization model missing at {settings.diarization_model_path}; "
                    "run `make models-local`")
    _ensure_asr_model(asr_model, settings.models_dir)
    settings.apply_library_env()

    from transcriber.hardware.detect import detect_hardware
    from transcriber.hardware.tiers import resolve_effective
    from transcriber.worker.runner import DefaultBackendFactory, JobRunner

    profile = detect_hardware(settings)
    eff = resolve_effective(settings, profile)
    factory = DefaultBackendFactory(settings, eff)
    llm_ok = factory.llm().status().usable

    client = TestClient(create_app(settings), headers={"X-API-Key": API_KEY})
    with open(video, "rb") as fh:
        resp = client.post("/v1/jobs", files={"file": (video.name, fh, "video/mp4")},
                           data={"cleanup": str(llm_ok).lower(),
                                 "summary": str(llm_ok).lower()})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["id"]

    conn = open_db(settings.db_path)
    runner = JobRunner(settings, conn, profile, eff, factory, threading.Event())
    claimed = repo.claim_next(conn, "e2e")
    assert claimed is not None
    assert runner.run(claimed) == "succeeded", repo.get_job(conn, job_id)

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["result"]["speakers"] >= 2

    doc = client.get(f"/v1/jobs/{job_id}/transcript").json()
    halves = [[u for u in doc["utterances"] if u["start"] < CLIP_SECONDS],
              [u for u in doc["utterances"] if u["start"] >= CLIP_SECONDS + 0.5]]
    reference = normalize_words(REFERENCE)
    for half in halves:
        hypothesis = normalize_words(" ".join(u["raw_text"] for u in half))
        assert substring_wer(reference, hypothesis) < 0.25, hypothesis

    for fmt in ("txt", "srt", "vtt"):
        body = client.get(f"/v1/jobs/{job_id}/transcript?format={fmt}").text
        assert body.strip()
    assert client.get(f"/v1/jobs/{job_id}/transcript?format=srt").text.startswith("1\n")
    assert client.get(f"/v1/jobs/{job_id}/transcript?format=vtt").text.startswith("WEBVTT")
    json.dumps(doc)
    if llm_ok:
        assert "## Overview" in client.get(f"/v1/jobs/{job_id}/summary").text


def test_substring_wer() -> None:
    ref = ["a", "b", "c"]
    assert substring_wer(ref, ["x", "y", "a", "b", "c", "z"]) == 0
    assert substring_wer(ref, ["a", "q", "c"]) == pytest.approx(1 / 3)
