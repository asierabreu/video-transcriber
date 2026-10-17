"""Real ffmpeg + SQLite queue + worker runner, with fake ML backends."""

import json
import threading

import pytest

from tests.api.test_api import submit
from tests.conftest import FakeFactory, FakeLlm, make_media, make_runner, needs_ffmpeg
from transcriber.store import jobs_repo as repo
from transcriber.store.files import JobFiles

pytestmark = [needs_ffmpeg, pytest.mark.integration]


def run_next(runner, conn):
    job = repo.claim_next(conn, "test-worker")
    assert job is not None
    return job["id"], runner.run(job)


@pytest.mark.parametrize("name", ["a.mp4", "a.mkv", "a.mov", "a.avi"])
def test_full_pipeline_every_container(client, settings, media_dir, name):
    job_id = submit(client, make_media(media_dir / name)).json()["id"]
    runner, conn = make_runner(settings, FakeFactory())
    assert run_next(runner, conn) == (job_id, "succeeded")

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == "succeeded" and job["progress"] == 1
    result = job["result"]
    assert result["speakers"] == 2 and result["words"] == 10
    assert result["cleanup_applied"] and result["summary_available"]
    assert result["formats"] == ["json", "txt", "srt", "vtt"]

    doc = client.get(f"/v1/jobs/{job_id}/transcript").json()
    assert doc["schema_version"] == 1
    assert [s["id"] for s in doc["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert doc["utterances"][0]["text"].startswith("Hello")
    txt = client.get(f"/v1/jobs/{job_id}/transcript?format=txt")
    assert txt.headers["content-type"].startswith("text/plain")
    assert "attachment" in txt.headers["content-disposition"]
    assert txt.text.startswith("[00:00:00] SPEAKER_00:")
    assert client.get(f"/v1/jobs/{job_id}/transcript?format=srt").text.startswith("1\n")
    assert client.get(f"/v1/jobs/{job_id}/transcript?format=vtt").text.startswith("WEBVTT")
    summary = client.get(f"/v1/jobs/{job_id}/summary")
    assert summary.status_code == 200 and "## Overview" in summary.text

    files = JobFiles(settings.jobs_dir, job_id)
    assert files.source() is None and not files.audio.exists()  # KEEP_MEDIA=false
    meta = files.read_json("meta.json")
    assert set(meta["result"]["stage_durations"]) >= {"extracting", "rendering"}


def test_options_disable_llm_steps(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4"), cleanup="false",
                    summary="false").json()["id"]
    llm = FakeLlm()
    runner, conn = make_runner(settings, FakeFactory(llm=llm))
    run_next(runner, conn)
    result = client.get(f"/v1/jobs/{job_id}").json()["result"]
    assert not result["cleanup_applied"] and not result["summary_available"]
    assert llm.calls == []
    assert client.get(f"/v1/jobs/{job_id}/summary").json()["error"]["code"] == \
        "summary_not_available"


def test_llm_unreachable_retries_then_degrades(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    runner, conn = make_runner(settings, FakeFactory(llm=FakeLlm(reachable=False)))
    assert run_next(runner, conn)[1] == "queued"
    job = repo.get_job(conn, job_id)
    assert job["not_before"] is not None and job["attempts"] == 1
    files = JobFiles(settings.jobs_dir, job_id)
    assert files.exists("aligned.json") and not files.exists("cleaned.json")

    # Last attempt: the job succeeds without post-processing.
    conn.execute("UPDATE jobs SET not_before = NULL, attempts = 2 WHERE id = ?", (job_id,))
    assert run_next(runner, conn)[1] == "succeeded"
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert not job["result"]["cleanup_applied"] and not job["result"]["summary_available"]
    assert any("cleanup skipped" in w for w in job["warnings"])


def test_model_not_pulled_degrades_immediately(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    runner, conn = make_runner(settings, FakeFactory(llm=FakeLlm(present=False)))
    assert run_next(runner, conn)[1] == "succeeded"
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["result"]["llm_fallbacks"] == 2  # both utterances kept their original text
    assert any("not pulled" in w for w in job["warnings"])


def test_resume_skips_completed_stages(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    factory = FakeFactory(diar_errors=[RuntimeError("boom")])  # CPU tier: one device to try
    runner, conn = make_runner(settings, factory)
    assert run_next(runner, conn)[1] == "queued"  # ResourceExhausted is transient
    conn.execute("UPDATE jobs SET not_before = NULL WHERE id = ?", (job_id,))
    asr_calls = len(factory.asr_attempts)
    assert run_next(runner, conn)[1] == "succeeded"
    assert len(factory.asr_attempts) == asr_calls  # transcription was not repeated


def test_corrupt_extraction_fails_permanently(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    source = JobFiles(settings.jobs_dir, job_id).source()
    source.write_bytes(b"\x00" * 100)
    runner, conn = make_runner(settings, FakeFactory())
    assert run_next(runner, conn)[1] == "failed"
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["error"]["code"] == "extraction_failed"


def test_cancel_running_job(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    runner, conn = make_runner(settings, FakeFactory())
    job = repo.claim_next(conn, "w")
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204  # running: flag only
    assert repo.get_job(conn, job_id)["status"] == "running"
    assert runner.run(job) == "cancelled"
    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "cancelled"
    assert not JobFiles(settings.jobs_dir, job_id).dir.exists()


def test_worker_stop_requeues_without_using_an_attempt(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    stop = threading.Event()
    stop.set()
    runner, conn = make_runner(settings, FakeFactory(), stop_event=stop)
    assert run_next(runner, conn)[1] == "queued"
    assert repo.get_job(conn, job_id)["attempts"] == 0


def test_unexpected_error_is_transient_once(client, settings, media_dir):
    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    factory = FakeFactory(asr_errors=[KeyError("bug"), KeyError("bug")])
    runner, conn = make_runner(settings, factory)
    assert run_next(runner, conn)[1] == "queued"
    conn.execute("UPDATE jobs SET not_before = NULL WHERE id = ?", (job_id,))
    assert run_next(runner, conn)[1] == "failed"
    assert repo.get_job(conn, job_id)["error"]["code"] == "internal_error"


def test_purge_removes_expired_jobs(client, settings, media_dir):
    from transcriber.worker.purge import cleanup_stale_files, purge_expired

    job_id = submit(client, make_media(media_dir / "a.mp4")).json()["id"]
    runner, conn = make_runner(settings, FakeFactory())
    run_next(runner, conn)
    conn.execute("UPDATE jobs SET finished_at = '2000-01-01T00:00:00Z'")
    orphan = settings.jobs_dir / "01JAB3K9ZQ4W1X8T6M2N5P7R0S"
    orphan.mkdir()
    assert purge_expired(conn, settings) == 1
    assert not (settings.jobs_dir / job_id).exists()
    assert cleanup_stale_files(conn, settings.jobs_dir, max_age_s=-1) == 1
    assert not orphan.exists()
    assert json.loads(json.dumps(repo.count_by_status(conn)))["succeeded"] == 0
