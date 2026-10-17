#!/usr/bin/env python3
"""One-time model download (docs/ARCHITECTURE.md §13.4). Needs internet access.

Downloads, into MODELS_DIR:
  whisper/<name>/                                   CTranslate2 Whisper models
  pyannote/speaker-diarization-community-1/         diarization pipeline (needs HF_TOKEN)
and pulls the Ollama model(s) through OLLAMA_URL, then writes MANIFEST.json.

Usage:
  python scripts/download_models.py --tier all
  python scripts/download_models.py --tier cuda-medium --llm-model llama3.1:8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Hugging Face libraries read this at import time.
os.environ["HF_HUB_OFFLINE"] = "0"

TIER_ASR = {
    "cuda-large": ["large-v3"],
    "cuda-medium": ["large-v3-turbo"],
    "cuda-small": ["distil-large-v3"],
    "apple": ["distil-large-v3"],
    "cpu": ["distil-large-v3"],
}
TIER_LLM = {"cuda-large": ["qwen3:8b"], "cuda-medium": ["qwen3:8b"], "cuda-small": ["qwen3:4b"],
            "apple": ["qwen3:8b"], "cpu": ["qwen3:4b"]}
ALL_ASR = ["large-v3", "large-v3-turbo", "distil-large-v3"]
ALL_LLM = ["qwen3:8b", "qwen3:4b"]
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"


def log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def sha256_dir(path: Path) -> dict[str, str]:
    """Checksums of every file in a directory, keyed by relative path."""
    out = {}
    for f in sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[str(f.relative_to(path))] = h.hexdigest()
    return out


def download_asr(models_dir: Path, names: list[str]) -> dict[str, dict[str, str]]:
    from faster_whisper import download_model

    result = {}
    for name in names:
        target = models_dir / "whisper" / name
        log(f"ASR model {name} -> {target}")
        download_model(name, output_dir=str(target))
        result[name] = sha256_dir(target)
    return result


def download_pyannote(models_dir: Path, token: str | None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    target = models_dir / "pyannote" / "speaker-diarization-community-1"
    if not token:
        sys.exit(
            "[models] HF_TOKEN is required to download the diarization model.\n"
            f"  1. Accept the conditions at https://huggingface.co/{PYANNOTE_REPO}\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Put HF_TOKEN=<token> in .env and run this again."
        )
    log(f"diarization pipeline {PYANNOTE_REPO} -> {target}")
    snapshot_download(repo_id=PYANNOTE_REPO, local_dir=str(target), token=token)
    if not (target / "config.yaml").is_file():
        sys.exit(f"[models] download finished but {target}/config.yaml is missing")
    return sha256_dir(target)


def pull_ollama(url: str, names: list[str]) -> None:
    import httpx

    base = url.rstrip("/")
    for attempt in range(30):
        try:
            httpx.get(f"{base}/api/version", timeout=5).raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 29:
                sys.exit(f"[models] Ollama is not reachable at {base}")
            time.sleep(2)
    for name in names:
        log(f"Ollama pull {name} (this can take a while)")
        resp = httpx.post(f"{base}/api/pull", json={"model": name, "stream": False},
                          timeout=httpx.Timeout(None, connect=10))
        if resp.status_code != 200 or resp.json().get("status") != "success":
            sys.exit(f"[models] pulling {name} failed: {resp.status_code} {resp.text[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="all", choices=["all", *TIER_ASR])
    parser.add_argument("--asr-model", action="append", help="extra/override ASR model names")
    parser.add_argument("--llm-model", action="append", help="extra/override Ollama models")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    models_dir = Path(os.environ.get("MODELS_DIR", "/models")).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    asr_names = args.asr_model or (ALL_ASR if args.tier == "all" else TIER_ASR[args.tier])
    env_llm = os.environ.get("LLM_MODEL") or None
    llm_names = args.llm_model or ([env_llm] if env_llm else
                                   (ALL_LLM if args.tier == "all" else TIER_LLM[args.tier]))

    manifest_path = models_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not args.skip_asr:
        manifest.setdefault("asr", {}).update(download_asr(models_dir, asr_names))
    if not args.skip_diarization:
        manifest["diarization"] = {PYANNOTE_REPO: download_pyannote(
            models_dir, os.environ.get("HF_TOKEN"))}
    if not args.skip_llm:
        pull_ollama(os.environ.get("OLLAMA_URL", "http://localhost:11434"), llm_names)
        manifest["llm"] = sorted(set(manifest.get("llm", [])) | set(llm_names))
    manifest["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log(f"done; manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
