"""/v1/jobs endpoints (§7.1)."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from transcriber.api.auth import require_api_key
from transcriber.api.deps import get_app_settings, get_conn
from transcriber.api.schemas import ErrorBody, Job, JobCreated, JobList, job_resource
from transcriber.api.uploads import stream_upload
from transcriber.config import Settings
from transcriber.errors import ApiError, MediaValidationError
from transcriber.ids import is_valid_job_id, new_job_id
from transcriber.pipeline.probe import run_ffprobe, validate_media
from transcriber.store import jobs_repo as repo
from transcriber.store.files import JobFiles, free_disk_gb

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])

ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorBody} for code in (400, 401, 404, 409, 413, 415, 422, 503)
}
MEDIA_TYPES = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")

UPLOAD_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {"multipart/form-data": {"schema": {
            "type": "object",
            "required": ["file"],
            "properties": {
                "file": {"type": "string", "format": "binary"},
                "num_speakers": {"type": "integer", "minimum": 1, "maximum": 20},
                "min_speakers": {"type": "integer", "minimum": 1, "maximum": 20},
                "max_speakers": {"type": "integer", "minimum": 1, "maximum": 20},
                "cleanup": {"type": "boolean", "default": True},
                "summary": {"type": "boolean", "default": True},
                "title": {"type": "string", "maxLength": 200},
            },
        }}},
    }
}


def _parse_bool(fields: dict[str, str], name: str, default: bool) -> bool:
    raw = fields.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ApiError(400, "invalid_options", f"'{name}' must be true or false.", {"field": name})


def _parse_speakers(fields: dict[str, str], name: str) -> int | None:
    raw = fields.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(400, "invalid_options", f"'{name}' must be an integer.",
                       {"field": name}) from None
    if not 1 <= value <= 20:
        raise ApiError(400, "invalid_options", f"'{name}' must be between 1 and 20.",
                       {"field": name})
    return value


def parse_options(fields: dict[str, str], filename: str | None) -> tuple[str, dict[str, Any]]:
    """Validate form fields into (title, options)."""
    options = {
        "num_speakers": _parse_speakers(fields, "num_speakers"),
        "min_speakers": _parse_speakers(fields, "min_speakers"),
        "max_speakers": _parse_speakers(fields, "max_speakers"),
        "cleanup": _parse_bool(fields, "cleanup", True),
        "summary": _parse_bool(fields, "summary", True),
    }
    lo, hi = options["min_speakers"], options["max_speakers"]
    if lo is not None and hi is not None and lo > hi:
        raise ApiError(400, "invalid_options", "'min_speakers' must not exceed 'max_speakers'.")
    title = (fields.get("title") or "").strip() or os.path.basename(filename or "") or "untitled"
    title = "".join(ch for ch in title if ch.isprintable())
    if len(title) > 200:
        raise ApiError(400, "invalid_options", "'title' must be at most 200 characters.")
    return title, options


def _download_name(title: str, ext: str) -> str:
    stem = re.sub(r"\.(mp4|mkv|mov|avi)$", "", title, flags=re.IGNORECASE)
    stem = _UNSAFE_FILENAME.sub("_", stem).strip(" ._") or "transcript"
    return f"{stem[:100]}.{ext}"


def _load(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    job = repo.get_job(conn, job_id) if is_valid_job_id(job_id) else None
    if job is None:
        raise ApiError(404, "not_found", f"Job '{job_id}' was not found.")
    return job


def _require_succeeded(job: dict[str, Any]) -> None:
    if job["status"] != "succeeded":
        raise ApiError(409, "job_not_ready", f"Job is {job['status']}; results are not available.",
                       {"status": job["status"]})


@router.post("", status_code=202, response_model=JobCreated, responses=ERRORS,
             openapi_extra=UPLOAD_OPENAPI, summary="Submit a video")
async def create_job(
    request: Request, settings: Settings = Depends(get_app_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> JSONResponse:
    """Stream the upload to disk, validate it with ffprobe, and queue a job."""
    if repo.count_by_status(conn)["queued"] >= settings.max_queued_jobs:
        raise ApiError(503, "queue_full", "Too many queued jobs; try again later.")
    declared = int(request.headers.get("content-length") or 0)
    if declared > settings.max_upload_bytes + 1024 * 1024:
        raise ApiError(413, "file_too_large",
                       f"File exceeds the limit of {settings.max_upload_mb} MB.",
                       {"max_upload_mb": settings.max_upload_mb})
    if free_disk_gb(settings.data_dir) < settings.min_free_disk_gb + declared / 2**30:
        raise ApiError(503, "insufficient_storage", "Not enough free disk space for this upload.")

    job_id = new_job_id()
    files = JobFiles(settings.jobs_dir, job_id)
    files.dir.mkdir(parents=True, exist_ok=False)
    try:
        upload = await stream_upload(request, files.upload_part, settings.max_upload_bytes)
        title, options = parse_options(upload.fields, upload.filename)
        try:
            probe = await run_in_threadpool(run_ffprobe, files.upload_part)
            media = validate_media(probe, settings.allowed_container_list,
                                   settings.max_duration_s, upload.size_bytes)
        except MediaValidationError as exc:
            raise exc.api_error from None
        os.replace(files.upload_part, files.path(f"source.{media['ext']}"))
        files.write_json("probe.json", probe)
        job = repo.create_job(conn, job_id, title, options, media)
    except BaseException:
        files.delete_all()
        raise
    log.info("job queued", extra={"job_id": job_id, "duration_s": media["duration_s"],
                                  "size_bytes": media["size_bytes"]})
    body = {
        "id": job["id"], "status": job["status"],
        "queue_position": repo.queue_position(conn, job), "created_at": job["created_at"],
        "links": {"self": f"/v1/jobs/{job['id']}"},
    }
    return JSONResponse(body, status_code=202)


@router.get("", response_model=JobList, responses=ERRORS, summary="List jobs")
def list_jobs(
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict[str, Any]:
    """List jobs newest first, paginated with an opaque cursor."""
    if cursor is not None and not is_valid_job_id(cursor):
        raise ApiError(400, "invalid_options", "Invalid cursor.")
    jobs, next_cursor = repo.list_jobs(conn, status, limit, cursor)
    return {"items": [job_resource(conn, j) for j in jobs], "next_cursor": next_cursor}


@router.get("/{job_id}", response_model=Job, responses=ERRORS, summary="Get a job")
def get_job(job_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
    """Return the job resource."""
    return job_resource(conn, _load(conn, job_id))


@router.get("/{job_id}/transcript", responses=ERRORS, summary="Download the transcript",
            response_class=FileResponse)
def get_transcript(
    job_id: str, format: Literal["json", "txt", "srt", "vtt"] = "json",
    settings: Settings = Depends(get_app_settings), conn: sqlite3.Connection = Depends(get_conn),
) -> FileResponse:
    """Serve a rendered transcript file."""
    job = _load(conn, job_id)
    _require_succeeded(job)
    path = JobFiles(settings.jobs_dir, job_id).output(format)
    if not path.is_file():
        raise ApiError(404, "not_found", "Transcript file is missing.")
    return FileResponse(path, media_type=MEDIA_TYPES[format],
                        filename=_download_name(job["title"], format))


@router.get("/{job_id}/summary", responses=ERRORS, summary="Download the summary",
            response_class=FileResponse)
def get_summary(
    job_id: str, settings: Settings = Depends(get_app_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> FileResponse:
    """Serve summary.md."""
    job = _load(conn, job_id)
    _require_succeeded(job)
    path = JobFiles(settings.jobs_dir, job_id).path("summary.md")
    if not path.is_file():
        raise ApiError(404, "summary_not_available",
                       "No summary for this job (disabled or the LLM was unavailable).")
    return FileResponse(path, media_type="text/markdown; charset=utf-8",
                        filename=_download_name(job["title"], "md"))


@router.delete("/{job_id}", status_code=204, responses=ERRORS, summary="Cancel or delete a job")
def delete_job(
    job_id: str, settings: Settings = Depends(get_app_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> Response:
    """Cancel a queued/running job, or delete a finished one, and remove its files.

    A running job stops at the next stage or chunk boundary; the worker then deletes its files.
    """
    job = _load(conn, job_id)
    files = JobFiles(settings.jobs_dir, job_id)
    if job["status"] in ("queued", "running"):
        if repo.cancel_if_queued(conn, job_id):
            files.delete_all()
        else:
            # Running (or claimed a moment ago): the worker cancels and cleans up.
            repo.request_cancel(conn, job_id)
    else:
        files.delete_all()
        repo.delete_job(conn, job_id)
    return Response(status_code=204)
