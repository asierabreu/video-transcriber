#!/usr/bin/env python3
"""Watch a queued or running transcription job with a tqdm progress bar.

Usage: API=http://localhost:8000 KEY=... python scripts/watch_job.py JOB_ID
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx
from tqdm import tqdm

STAGES = (
    "extracting",
    "transcribing",
    "diarizing",
    "aligning",
    "cleaning",
    "summarizing",
    "rendering",
)
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a transcription job.")
    parser.add_argument("job", nargs="?", default=os.environ.get("JOB"), help="job ID")
    parser.add_argument("--url", default=os.environ.get("API", "http://localhost:8000"),
                        help="API base URL (default: API or http://localhost:8000)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="seconds between status checks (default: 2)")
    args = parser.parse_args()

    key = os.environ.get("KEY") or os.environ.get("API_KEY")
    if not key:
        parser.error("set KEY or API_KEY")
    if not args.job:
        parser.error("provide a job ID or set JOB")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    endpoint = f"/v1/jobs/{args.job}"
    headers = {"X-API-Key": key}
    with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=10) as client:
        try:
            job = _get_job(client, endpoint)
        except httpx.HTTPError as exc:
            sys.exit(f"request failed: {exc}")

        title = job.get("title", args.job)
        with tqdm(total=100, unit="%", bar_format="{l_bar}{bar}| {n:5.1f}% {postfix}",
                  desc=title[:32], file=sys.stdout) as progress:
            while True:
                _update_bar(progress, job)
                if job["status"] in TERMINAL_STATUSES:
                    break
                time.sleep(args.interval)
                try:
                    job = _get_job(client, endpoint)
                except httpx.HTTPError as exc:
                    progress.write(f"request failed: {exc}")

            progress.close()
            _print_result(job)

        if job["status"] != "succeeded":
            sys.exit(1)


def _get_job(client: httpx.Client, endpoint: str) -> dict[str, object]:
    response = client.get(endpoint)
    response.raise_for_status()
    return response.json()


def _update_bar(progress: tqdm, job: dict[str, object]) -> None:
    status = str(job.get("status", "unknown"))
    stage = str(job.get("stage") or "")
    stage_label = stage or status
    percent = max(0.0, min(100.0, float(job.get("progress", 0.0)) * 100))
    completed = STAGES.index(stage) if stage in STAGES else len(STAGES) if status == "succeeded" else 0
    stage_list = " ".join(
        f"{'>' if name == stage else '✓' if STAGES.index(name) < completed else '·'}{name[:4]}"
        for name in STAGES
    )
    queue = job.get("queue_position")
    queue_text = f" queue={queue}" if status == "queued" and queue else ""
    progress.n = percent
    progress.set_postfix_str(f"{status} | {stage_label} | {stage_list}{queue_text}")
    progress.refresh()


def _print_result(job: dict[str, object]) -> None:
    status = job["status"]
    if status == "succeeded":
        result = job.get("result") or {}
        print(f"completed: {result.get('processing_s', 0):.1f}s processing")
    elif status == "failed":
        print(f"failed: {job.get('error')}", file=sys.stderr)
    else:
        print(f"job {status}")


if __name__ == "__main__":
    main()
