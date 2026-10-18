#!/usr/bin/env python3
"""Launch video-transcriber jobs and service health commands from the command line.

Usage: API=http://localhost:8000 KEY=... ./scripts/transcriber.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    if exc.name in {"httpx", "tqdm"}:
        raise SystemExit(
            "Missing CLI dependencies. Install the project first with: "
            "python3 -m pip install -e ."
        ) from None
    raise

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
FORMATS = ("json", "txt", "srt", "vtt", "summary")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    key = os.environ.get("KEY") or os.environ.get("API_KEY")
    if not key:
        parser.error("set KEY or API_KEY")

    with httpx.Client(base_url=args.url.rstrip("/"),
                      headers={"X-API-Key": key}, timeout=60) as client:
        try:
            result = _dispatch(client, args)
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc.response)
            raise SystemExit(1) from None
        except httpx.HTTPError as exc:
            raise SystemExit(f"request failed: {exc}") from None

    if result is not None:
        _print_result(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage video-transcriber jobs and services.")
    parser.add_argument("--url", default=os.environ.get("API", "http://localhost:8000"),
                        help="API base URL (default: API or http://localhost:8000)")
    subparsers = parser.add_subparsers(dest="command")

    submit = subparsers.add_parser("submit", help="submit a video for transcription")
    submit.add_argument("video", type=Path)
    submit.add_argument("--num-speakers", type=int)
    submit.add_argument("--min-speakers", type=int)
    submit.add_argument("--max-speakers", type=int)
    submit.add_argument("--diarization", action=argparse.BooleanOptionalAction, default=False)
    submit.add_argument("--title")
    submit.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=False)
    submit.add_argument("--summary", action=argparse.BooleanOptionalAction, default=False)

    watch = subparsers.add_parser("watch", help="watch a job with a progress bar")
    watch.add_argument("job", nargs="?", default=os.environ.get("JOB"))
    watch.add_argument("--interval", type=float, default=2.0)

    job = subparsers.add_parser("job", help="show one job")
    job.add_argument("job", nargs="?", default=os.environ.get("JOB"))

    jobs = subparsers.add_parser("list", help="list jobs")
    jobs.add_argument("--status", choices=("queued", "running", "succeeded", "failed", "cancelled"))
    jobs.add_argument("--limit", type=int, default=20)
    jobs.add_argument("--cursor")

    download = subparsers.add_parser("download", help="download a transcript or summary")
    download.add_argument("job", nargs="?", default=os.environ.get("JOB"))
    download.add_argument("--format", choices=FORMATS, default="json")
    download.add_argument("--output", type=Path)

    delete = subparsers.add_parser("delete", help="cancel or delete a job")
    delete.add_argument("job", nargs="?", default=os.environ.get("JOB"))

    subparsers.add_parser("system", help="show system, model, worker, and queue information")
    subparsers.add_parser("health", help="show liveness health")
    subparsers.add_parser("ready", help="show readiness checks")
    subparsers.add_parser("metrics", help="show Prometheus metrics")
    return parser


def _dispatch(client: httpx.Client, args: argparse.Namespace) -> Any:
    if args.command == "submit":
        return _submit(client, args)
    if args.command == "watch":
        _watch(client, args.job, args.interval)
        return None
    if args.command == "job":
        return _get_json(client, f"/v1/jobs/{_required_job(args.job)}")
    if args.command == "list":
        if not 1 <= args.limit <= 100:
            raise SystemExit("--limit must be between 1 and 100")
        params = {key: value for key, value in {
            "status": args.status, "limit": args.limit, "cursor": args.cursor,
        }.items() if value is not None}
        return _get_json(client, "/v1/jobs", params=params)
    if args.command == "download":
        return _download(client, args)
    if args.command == "delete":
        response = client.delete(f"/v1/jobs/{_required_job(args.job)}")
        response.raise_for_status()
        print("job cancelled or deleted")
        return None
    paths = {"system": "/v1/system", "health": "/healthz", "ready": "/readyz"}
    if args.command in paths:
        allowed = {200, 503} if args.command == "ready" else {200}
        return _get_json(client, paths[args.command], allowed_statuses=allowed)
    if args.command == "metrics":
        response = client.get("/metrics")
        response.raise_for_status()
        print(response.text, end="")
        return None
    raise SystemExit(f"unknown command: {args.command}")


def _submit(client: httpx.Client, args: argparse.Namespace) -> dict[str, Any]:
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    data = {key: value for key, value in {
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "diarization": str(args.diarization).lower(),
        "cleanup": str(args.cleanup).lower(),
        "summary": str(args.summary).lower(),
        "title": args.title,
    }.items() if value is not None}
    with args.video.open("rb") as video:
        response = client.post("/v1/jobs", files={"file": (args.video.name, video)}, data=data)
    response.raise_for_status()
    return response.json()


def _watch(client: httpx.Client, job_id: str | None, interval: float) -> None:
    job_id = _required_job(job_id)
    if interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    job = _get_json(client, f"/v1/jobs/{job_id}")
    with tqdm(total=100, unit="%", bar_format="{l_bar}{bar}| {n:5.1f}% {postfix}",
              desc=str(job.get("title", job_id))[:32], file=sys.stdout) as progress:
        while True:
            _update_progress(progress, job)
            if job["status"] in TERMINAL_STATUSES:
                break
            time.sleep(interval)
            job = _get_json(client, f"/v1/jobs/{job_id}")
    if job["status"] == "succeeded":
        result = job.get("result") or {}
        print(f"completed: {result.get('processing_s', 0):.1f}s processing")
    elif job["status"] == "failed":
        print(f"failed: {job.get('error')}", file=sys.stderr)
        raise SystemExit(1)
    else:
        print(f"job {job['status']}")


def _update_progress(progress: tqdm, job: dict[str, Any]) -> None:
    status = str(job.get("status", "unknown"))
    stage = str(job.get("stage") or "")
    label = stage or status
    percent = max(0.0, min(100.0, float(job.get("progress", 0.0)) * 100))
    completed = STAGES.index(stage) if stage in STAGES else len(STAGES) if status == "succeeded" else 0
    stages = " ".join(
        f"{'>' if name == stage else '✓' if STAGES.index(name) < completed else '·'}{name[:4]}"
        for name in STAGES
    )
    queue = job.get("queue_position")
    queue_text = f" queue={queue}" if status == "queued" and queue else ""
    progress.n = percent
    progress.set_postfix_str(f"{status} | {label} | {stages}{queue_text}")
    progress.refresh()


def _download(client: httpx.Client, args: argparse.Namespace) -> None:
    job_id = _required_job(args.job)
    extension = "md" if args.format == "summary" else args.format
    path = args.output or Path(f"{job_id}.{extension}")
    endpoint = f"/v1/jobs/{job_id}/summary" if args.format == "summary" else \
        f"/v1/jobs/{job_id}/transcript?format={args.format}"
    response = client.get(endpoint)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"saved {path}")


def _get_json(client: httpx.Client, endpoint: str, *, allowed_statuses: set[int] | None = None,
              **kwargs: Any) -> Any:
    response = client.get(endpoint, **kwargs)
    if response.status_code not in (allowed_statuses or {200}):
        response.raise_for_status()
    return response.json()


def _required_job(job: str | None) -> str:
    if not job:
        raise SystemExit("provide a job ID or set JOB")
    return job


def _print_result(result: Any) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _print_http_error(response: httpx.Response) -> None:
    try:
        print(json.dumps(response.json(), indent=2), file=sys.stderr)
    except ValueError:
        print(f"HTTP {response.status_code}: {response.text}", file=sys.stderr)


if __name__ == "__main__":
    main()
