"""Prometheus metrics computed from SQLite at scrape time (§11)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
    Metric,
)
from prometheus_client.registry import Collector

from transcriber.api.routes.system import worker_status
from transcriber.store import jobs_repo as repo
from transcriber.store.db import connect, now_iso

router = APIRouter(tags=["metrics"])

JOB_BUCKETS = [60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, 7200.0, 14400.0]
STAGE_BUCKETS = [5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 3600.0]


def _histogram_buckets(values: list[float], bounds: list[float]) -> list[tuple[str, float]]:
    buckets = [(str(float(b)), float(sum(1 for v in values if v <= b))) for b in bounds]
    buckets.append(("+Inf", float(len(values))))
    return buckets


class JobsCollector(Collector):
    """Reads job state for every scrape, so API and worker need no shared metric state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def collect(self) -> Iterator[Metric]:
        conn = connect(self.db_path)
        try:
            counts = repo.count_by_status(conn)
            jobs = CounterMetricFamily("transcribe_jobs", "Jobs by status.", labels=["status"])
            for status in ("succeeded", "failed", "cancelled"):
                jobs.add_metric([status], counts[status])
            yield jobs

            yield GaugeMetricFamily("transcribe_queue_depth", "Queued jobs.",
                                    value=counts["queued"])

            recent = repo.finished_jobs_since(conn, now_iso(-86400))
            durations = [float(j["result"].get("processing_s", 0)) for j in recent]
            hist = HistogramMetricFamily("transcribe_job_duration_seconds",
                                         "Processing time of jobs finished in the last 24 h.")
            hist.add_metric([], _histogram_buckets(durations, JOB_BUCKETS), sum(durations))
            yield hist

            stage_hist = HistogramMetricFamily(
                "transcribe_stage_duration_seconds",
                "Stage durations of jobs finished in the last 24 h.", labels=["stage"])
            by_stage: dict[str, list[float]] = {}
            for job in recent:
                for stage, secs in (job["result"].get("stage_durations") or {}).items():
                    by_stage.setdefault(stage, []).append(float(secs))
            for stage, values in sorted(by_stage.items()):
                stage_hist.add_metric([stage], _histogram_buckets(values, STAGE_BUCKETS),
                                      sum(values))
            yield stage_hist

            rtf = 0.0
            if recent:
                last = recent[-1]
                duration = float(last["media"].get("duration_s") or 0)
                if duration > 0:
                    rtf = float(last["result"].get("processing_s", 0)) / duration
            yield GaugeMetricFamily("transcribe_realtime_factor",
                                    "Processing time / media duration of the last job.",
                                    value=round(rtf, 4))

            yield GaugeMetricFamily("transcribe_worker_up", "1 if the worker heartbeat is fresh.",
                                    value=1.0 if worker_status(conn)["alive"] else 0.0)

            fallbacks = CounterMetricFamily(
                "transcribe_llm_fallbacks",
                "Utterances that kept their original text (guard rejections, LLM errors).")
            fallbacks.add_metric([], repo.total_llm_fallbacks(conn))
            yield fallbacks
        finally:
            conn.close()


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    """Expose metrics in the Prometheus text format."""
    registry = CollectorRegistry(auto_describe=False)
    registry.register(JobsCollector(request.app.state.settings.db_path))
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
