"""Response models (used for validation and the OpenAPI document) and serializers."""

from __future__ import annotations

import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, Field

from transcriber.store import jobs_repo as repo

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class ErrorObject(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error: ErrorObject


class JobOptions(BaseModel):
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization: bool = False
    cleanup: bool = False
    summary: bool = False


class MediaInfo(BaseModel):
    container: str
    duration_s: float
    size_bytes: int


class JobLinks(BaseModel):
    self: str


class JobCreated(BaseModel):
    id: str
    status: JobStatus
    queue_position: int | None
    created_at: str
    links: JobLinks


class Job(BaseModel):
    id: str
    title: str
    status: JobStatus
    stage: str | None
    progress: float
    queue_position: int | None
    options: JobOptions
    media: MediaInfo
    result: dict[str, Any] | None
    warnings: list[str]
    error: ErrorObject | None
    attempts: int
    created_at: str
    started_at: str | None
    finished_at: str | None


class JobList(BaseModel):
    items: list[Job]
    next_cursor: str | None


def job_resource(conn: sqlite3.Connection, job: dict[str, Any]) -> dict[str, Any]:
    """Serialize a job row into the public job resource (§7.1)."""
    media = job["media"]
    return {
        "id": job["id"],
        "title": job["title"],
        "status": job["status"],
        "stage": job["stage"] if job["status"] == "running" else None,
        "progress": job["progress"],
        "queue_position": repo.queue_position(conn, job),
        "options": job["options"],
        "media": {"container": media["container"], "duration_s": media["duration_s"],
                  "size_bytes": media["size_bytes"]},
        "result": job["result"] if job["status"] == "succeeded" else None,
        "warnings": job["warnings"],
        "error": job["error"] if job["status"] in ("failed", "cancelled") else None,
        "attempts": job["attempts"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }
