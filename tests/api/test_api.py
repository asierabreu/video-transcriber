import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_media, needs_ffmpeg
from transcriber.api.main import create_app
from transcriber.config import Settings
from transcriber.store import jobs_repo as repo
from transcriber.store.db import open_db


def submit(client, path, **fields):
    with open(path, "rb") as fh:
        return client.post("/v1/jobs", files={"file": (path.name, fh, "video/mp4")},
                           data={k: str(v) for k, v in fields.items()})


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code


def test_auth_required(client):
    resp = client.get("/v1/jobs", headers={"X-API-Key": "wrong"})
    assert_error(resp, 401, "unauthorized")
    resp = TestClient(client.app).get("/v1/system")
    assert_error(resp, 401, "unauthorized")
    assert TestClient(client.app).get("/healthz").status_code == 200


def test_api_refuses_to_start_with_short_key(settings):
    with pytest.raises(ValueError):
        create_app(settings.model_copy(update={"api_key": "short"}))


def test_unknown_job_and_bad_ids(client):
    assert_error(client.get("/v1/jobs/01JAB3K9ZQ4W1X8T6M2N5P7R0S"), 404, "not_found")
    assert_error(client.get("/v1/jobs/..%2F..%2Fetc"), 404, "not_found")
    assert_error(client.get("/v1/jobs?cursor=nope"), 400, "invalid_options")
    assert_error(client.get("/v1/jobs?limit=1000"), 400, "invalid_options")
    assert_error(client.get("/nope"), 404, "not_found")


def test_request_id_echoed(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["x-request-id"] == "abc-123"


def test_non_multipart_rejected(client):
    assert_error(client.post("/v1/jobs", content=b"x"), 400, "invalid_options")


@needs_ffmpeg
def test_submit_get_list_and_delete(client, settings, media_dir):
    resp = submit(client, make_media(media_dir / "talk.mp4"), num_speakers=2, summary="false")
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "queued" and body["queue_position"] == 1
    job_id = body["id"]

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["title"] == "talk.mp4"
    assert job["options"] == {"num_speakers": 2, "min_speakers": None, "max_speakers": None,
                              "cleanup": True, "summary": False}
    assert job["media"]["duration_s"] > 2 and "mp4" in job["media"]["container"]
    assert (settings.jobs_dir / job_id / "source.mp4").exists()
    assert (settings.jobs_dir / job_id / "probe.json").exists()

    assert client.get("/v1/jobs").json()["items"][0]["id"] == job_id
    assert client.get("/v1/jobs?status=failed").json()["items"] == []
    assert_error(client.get(f"/v1/jobs/{job_id}/transcript"), 409, "job_not_ready")
    assert_error(client.get(f"/v1/jobs/{job_id}/summary"), 409, "job_not_ready")

    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "cancelled"
    assert not (settings.jobs_dir / job_id).exists()
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204  # finished: row removed
    assert_error(client.get(f"/v1/jobs/{job_id}"), 404, "not_found")


@needs_ffmpeg
@pytest.mark.parametrize("name", ["clip.mkv", "clip.mov", "clip.avi"])
def test_all_allowed_containers_accepted(client, media_dir, name):
    resp = submit(client, make_media(media_dir / name))
    assert resp.status_code == 202, resp.text


@needs_ffmpeg
def test_validation_errors(client, settings, media_dir):
    assert_error(submit(client, make_media(media_dir / "a.wav", video=False)), 415,
                 "unsupported_media")
    assert_error(submit(client, make_media(media_dir / "silent.mp4", audio=False)), 422,
                 "no_audio_stream")
    junk = media_dir / "junk.mp4"
    junk.write_bytes(b"\x00" * 2048)
    assert_error(submit(client, junk), 422, "unreadable_media")
    good = make_media(media_dir / "ok.mp4")
    assert_error(submit(client, good, min_speakers=3, max_speakers=2), 400, "invalid_options")
    assert_error(submit(client, good, cleanup="maybe"), 400, "invalid_options")
    assert_error(submit(client, good, num_speakers=99), 400, "invalid_options")
    # Nothing is left behind by rejected uploads.
    assert list(settings.jobs_dir.iterdir()) == []


@needs_ffmpeg
def test_duration_and_size_limits(settings, media_dir):
    path = make_media(media_dir / "long.mp4", seconds=3)
    small = settings.model_copy(update={"max_duration_s": 1})
    client = TestClient(create_app(small), headers={"X-API-Key": settings.api_key})
    assert_error(submit(client, path), 422, "duration_exceeded")
    tiny: Settings = settings.model_copy(update={"max_upload_mb": 1})
    client = TestClient(create_app(tiny), headers={"X-API-Key": settings.api_key})
    big = media_dir / "big.mp4"
    big.write_bytes(b"\x00" * (3 * 1024 * 1024))
    assert_error(submit(client, big), 413, "file_too_large")


@needs_ffmpeg
def test_queue_full(settings, media_dir):
    limited = settings.model_copy(update={"max_queued_jobs": 1})
    client = TestClient(create_app(limited), headers={"X-API-Key": settings.api_key})
    path = make_media(media_dir / "q.mp4")
    assert submit(client, path).status_code == 202
    assert_error(submit(client, path), 503, "queue_full")


def test_system_readyz_and_metrics(client, settings):
    conn = open_db(settings.db_path)
    sys_info = client.get("/v1/system").json()
    assert sys_info["worker"]["alive"] is False and sys_info["queue"] == {"queued": 0,
                                                                         "running": 0}
    ready = client.get("/readyz")
    assert ready.status_code == 503 and ready.json()["checks"]["worker"]["ok"] is False
    assert ready.json()["checks"]["ollama"]["ok"] is False

    repo.set_system(conn, "worker_heartbeat", {"worker_id": "w", "current_job": None})
    repo.set_system(conn, "worker_profile", {"hardware": {"device": "cpu", "tier": "cpu"},
                                             "models": {"asr_model": "distil-large-v3"}})
    assert client.get("/readyz").status_code == 200
    sys_info = client.get("/v1/system").json()
    assert sys_info["hardware"]["tier"] == "cpu" and sys_info["worker"]["alive"]

    text = client.get("/metrics").text
    for name in ("transcribe_jobs_total", "transcribe_queue_depth",
                 "transcribe_job_duration_seconds_bucket", "transcribe_worker_up 1.0",
                 "transcribe_realtime_factor", "transcribe_llm_fallbacks_total"):
        assert name in text


def test_docs_disabled_by_default(client, settings):
    assert client.get("/docs").status_code == 404
    docs_client = TestClient(create_app(settings.model_copy(update={"docs_enabled": True})))
    assert docs_client.get("/openapi.json").status_code == 200
