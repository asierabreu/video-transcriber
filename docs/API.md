# API reference

Base URL: `http://<host>:8000`. Every `/v1` endpoint requires `X-API-Key: <API_KEY>`.
`/healthz`, `/readyz` and `/metrics` need no key. Every response carries an `X-Request-ID`
header; send your own to correlate logs.

## Errors

Every non-2xx response has this shape:

```json
{ "error": { "code": "unsupported_media", "message": "Container 'flv' is not supported. Allowed: mp4, mkv, mov, avi.", "details": { "detected": "flv" } } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `unauthorized` | 401 | Missing or invalid API key |
| `not_found` | 404 | Unknown job ID or route |
| `invalid_options` | 400 | Bad form fields or query parameters, malformed multipart body |
| `file_too_large` | 413 | Upload exceeds `MAX_UPLOAD_MB` |
| `unsupported_media` | 415 | Container is not mp4, mkv, mov or avi |
| `no_audio_stream` / `duration_exceeded` / `unreadable_media` | 422 | ffprobe validation failed |
| `job_not_ready` | 409 | Results requested before the job succeeded |
| `summary_not_available` | 404 | Summary disabled for the job, or the LLM was unavailable |
| `queue_full` | 503 | `MAX_QUEUED_JOBS` queued jobs already |
| `insufficient_storage` | 503 | Free disk below `MIN_FREE_DISK_GB` plus the upload size |
| `internal_error` | 500 | Unexpected; `details.request_id` matches the server logs |

A failed job carries one of these codes in `job.error.code`: `extraction_failed`,
`transcription_failed`, `diarization_failed`, `model_unavailable`, `out_of_resources`,
`llm_unavailable`, `internal_error`. A cancelled job carries `cancelled`.

## `POST /v1/jobs`

`multipart/form-data`. The response is **202**.

| Field | Type | Default |
|---|---|---|
| `file` | binary, required | — |
| `num_speakers` | int 1–20 | none |
| `min_speakers`, `max_speakers` | int 1–20 | none |
| `cleanup` | `true`/`false` | `true` |
| `summary` | `true`/`false` | `true` |
| `title` | string ≤ 200 | the uploaded filename |

The upload streams to disk and is checked with ffprobe before the response is sent, so invalid
files fail immediately.

```json
{ "id": "01JAB3K9ZQ4W1X8T6M2N5P7R0S", "status": "queued", "queue_position": 2,
  "created_at": "2026-10-17T18:12:03Z", "links": { "self": "/v1/jobs/01JAB3K9ZQ4W1X8T6M2N5P7R0S" } }
```

## `GET /v1/jobs`

Query parameters: `status` (`queued|running|succeeded|failed|cancelled`), `limit` (1–100,
default 20), `cursor` (the `next_cursor` from the previous page). Returns
`{ "items": [Job], "next_cursor": "..." | null }`, newest first.

## `GET /v1/jobs/{id}`

```json
{
  "id": "01JAB3...", "title": "weekly-sync.mp4", "status": "running",
  "stage": "diarizing", "progress": 0.56, "queue_position": null,
  "options": { "num_speakers": null, "min_speakers": null, "max_speakers": null, "cleanup": true, "summary": true },
  "media": { "container": "mov,mp4,m4a,3gp,3g2,mj2", "duration_s": 3612.4, "size_bytes": 812349120 },
  "result": null, "warnings": [], "error": null, "attempts": 1,
  "created_at": "...", "started_at": "...", "finished_at": null
}
```

- `stage` is one of `extracting`, `transcribing`, `diarizing`, `aligning`, `cleaning`,
  `summarizing` or `rendering`, and is set only while the job is running.
- `warnings` lists degraded behavior, such as a model fallback, LLM cleanup that was skipped, or
  utterances that kept their original text.
- When `status` is `succeeded`, `result` is:

```json
{ "speakers": 3, "words": 9412, "cleanup_applied": true, "summary_available": true,
  "formats": ["json","txt","srt","vtt"],
  "models": {"asr": "large-v3", "diarization": "community-1", "llm": "qwen3:8b"},
  "llm_fallbacks": 4, "processing_s": 412.3,
  "stage_durations": {"extracting": 6.1, "transcribing": 190.4, "...": 0} }
```

## `GET /v1/jobs/{id}/transcript?format=json|txt|srt|vtt`

The default format is `json`. The file is sent as an attachment. Content types are
`application/json`, `text/plain; charset=utf-8`, `application/x-subrip` and `text/vtt`.
Requesting a transcript before the job succeeds returns **409** `job_not_ready`.

### `transcript.json` (schema v1)

```json
{
  "schema_version": 1, "job_id": "01JAB3...", "language": "en", "duration_s": 3612.4,
  "models": { "asr": "large-v3", "diarization": "pyannote/speaker-diarization-community-1", "llm": "qwen3:8b" },
  "cleanup_applied": true,
  "speakers": [ { "id": "SPEAKER_00", "talk_time_s": 1402.3, "words": 3811 } ],
  "utterances": [ {
      "id": "u_0000", "speaker": "SPEAKER_00", "start": 0.84, "end": 7.42,
      "text": "Good morning, everyone. Let's start with the roadmap.",
      "raw_text": "good morning everyone lets start with the road map",
      "confidence": 0.91,
      "words": [ { "w": "good", "start": 0.84, "end": 1.02, "p": 0.97, "speaker": "SPEAKER_00" } ]
  } ],
  "warnings": []
}
```

`text` is the cleaned text and `raw_text` is the unedited ASR output. Word timings always come
from the ASR.

### TXT, SRT, VTT

- **TXT** has one paragraph per speaker turn: `[00:00:00] SPEAKER_00: Good morning, …`.
- **SRT** cues are at most 7 s long and 84 characters, on two lines, prefixed with
  `[SPEAKER_00] `.
- **VTT** uses voice tags: `<v SPEAKER_00>Good morning, …`.

## `GET /v1/jobs/{id}/summary`

Returns `text/markdown` with the sections `## Overview`, `## Key points`, `## Decisions`,
`## Action items` and `## Topics`.

## `DELETE /v1/jobs/{id}`

Returns **204**.

- **Queued job:** it is cancelled and its files are deleted.
- **Running job:** it is flagged for cancellation. The worker stops at the next stage or chunk
  boundary, then deletes the files. The job ends up `cancelled`.
- **Finished job:** its files and its record are deleted.

## `GET /v1/system`

```json
{ "version": "1.0.0",
  "hardware": {"device": "cuda", "tier": "cuda-medium", "gpu": "NVIDIA RTX 4070", "vram_gb": 12.0, "cpu_threads": 16, "notes": []},
  "models": {"tier": "cuda-medium", "asr_model": "large-v3-turbo", "asr_compute_type": "int8_float16", "...": "..."},
  "worker": {"alive": true, "last_heartbeat": "2026-10-17T18:20:01Z", "current_job": "01JAB3..."},
  "queue": {"queued": 1, "running": 1} }
```

## `GET /healthz`, `GET /readyz`, `GET /metrics`

- `/healthz` returns `{"status": "ok"}`.
- `/readyz` returns 200 or 503 with `{"status": "ready"|"not_ready", "checks": {...}}`. It
  checks that the database is writable, that free disk is at least `MIN_FREE_DISK_GB`, and
  that the worker heartbeat is less than 60 s old. Ollama's status is included for
  information only.
- `/metrics` exposes `transcribe_jobs_total{status}`, `transcribe_queue_depth`,
  `transcribe_job_duration_seconds`, `transcribe_stage_duration_seconds{stage}`,
  `transcribe_realtime_factor`, `transcribe_worker_up` and `transcribe_llm_fallbacks_total`.
