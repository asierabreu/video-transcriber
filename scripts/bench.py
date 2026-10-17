#!/usr/bin/env python3
"""Submit one video to a running service and report the real-time factor per stage.

Usage: API_KEY=... python scripts/bench.py recording.mp4 [--url http://localhost:8000]
Record the output in docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--no-llm", action="store_true", help="disable cleanup and summary")
    args = parser.parse_args()
    key = os.environ.get("API_KEY")
    if not key:
        sys.exit("set API_KEY")
    client = httpx.Client(base_url=args.url, headers={"X-API-Key": key},
                          timeout=httpx.Timeout(600, connect=10))

    system = client.get("/v1/system").json()
    with open(args.video, "rb") as fh:
        data = {"cleanup": "false", "summary": "false"} if args.no_llm else {}
        resp = client.post("/v1/jobs", files={"file": (args.video.name, fh)}, data=data)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"job {job_id} submitted; tier={system['hardware']['tier']}")

    while True:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        print(f"  {job['status']:<9} {job['stage'] or '':<12} {job['progress'] * 100:5.1f}%")
        time.sleep(10)
    if job["status"] != "succeeded":
        sys.exit(f"job {job['status']}: {job['error']}")

    duration = job["media"]["duration_s"]
    result = job["result"]
    print(f"\nmedia {duration / 60:.1f} min, processing {result['processing_s'] / 60:.1f} min, "
          f"RTF {result['processing_s'] / duration:.3f}")
    print(f"models: {result['models']}")
    for stage, secs in result["stage_durations"].items():
        print(f"  {stage:<13} {secs:8.1f} s   RTF {secs / duration:.3f}")


if __name__ == "__main__":
    main()
