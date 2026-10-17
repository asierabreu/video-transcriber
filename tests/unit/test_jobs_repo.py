import time

from transcriber.ids import new_job_id
from transcriber.store import jobs_repo as repo
from transcriber.store.db import open_db

MEDIA = {"container": "avi", "ext": "avi", "duration_s": 1.0, "size_bytes": 1}


def new(conn, title="t"):
    job_id = new_job_id()
    repo.create_job(conn, job_id, title, {"cleanup": True}, MEDIA)
    time.sleep(0.002)  # distinct ULID timestamps keep ordering deterministic
    return job_id


def test_claim_is_fifo_and_increments_attempts(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a, b = new(conn), new(conn)
    assert repo.queue_position(conn, repo.get_job(conn, b)) == 2
    job = repo.claim_next(conn, "w1")
    assert job["id"] == a and job["status"] == "running" and job["attempts"] == 1
    assert repo.claim_next(conn, "w1")["id"] == b
    assert repo.claim_next(conn, "w1") is None


def test_requeue_with_delay_is_not_claimable(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a = new(conn)
    repo.claim_next(conn, "w")
    repo.requeue(conn, a, 60)
    assert repo.claim_next(conn, "w") is None
    repo.requeue(conn, a, 0, refund_attempt=True)
    assert repo.get_job(conn, a)["attempts"] == 0
    assert repo.claim_next(conn, "w")["id"] == a


def test_cancel_if_queued(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a, b = new(conn), new(conn)
    assert repo.cancel_if_queued(conn, a)
    assert repo.get_job(conn, a)["status"] == "cancelled"
    repo.claim_next(conn, "w")  # claims b; a is cancelled
    assert not repo.cancel_if_queued(conn, b)


def test_recover_stale(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a = new(conn)
    repo.claim_next(conn, "w")
    conn.execute("UPDATE jobs SET heartbeat_at = '2000-01-01T00:00:00Z' WHERE id = ?", (a,))
    assert repo.recover_stale(conn, 300, max_attempts=3) == [a]
    assert repo.get_job(conn, a)["status"] == "queued"
    repo.claim_next(conn, "w")
    conn.execute("UPDATE jobs SET heartbeat_at = '2000-01-01T00:00:00Z', attempts = 3")
    repo.recover_stale(conn, 300, max_attempts=3)
    assert repo.get_job(conn, a)["status"] == "failed"


def test_list_pagination_and_expiry(tmp_path):
    conn = open_db(tmp_path / "j.db")
    ids = [new(conn) for _ in range(5)]
    page1, cursor = repo.list_jobs(conn, None, 2, None)
    assert [j["id"] for j in page1] == ids[::-1][:2] and cursor == ids[3]
    page2, _ = repo.list_jobs(conn, None, 2, cursor)
    assert [j["id"] for j in page2] == [ids[2], ids[1]]
    conn.execute("UPDATE jobs SET status='succeeded', finished_at='2000-01-01T00:00:00Z' "
                 "WHERE id = ?", (ids[0],))
    assert repo.expired_job_ids(conn, 2) == [ids[0]]
    assert repo.expired_job_ids(conn, 0) == []


def test_warnings_deduplicated_and_system_table(tmp_path):
    conn = open_db(tmp_path / "j.db")
    a = new(conn)
    repo.add_warnings(conn, a, ["x", "y"])
    repo.add_warnings(conn, a, ["x"])
    assert repo.get_job(conn, a)["warnings"] == ["x", "y"]
    repo.set_system(conn, "k", {"v": 1})
    assert repo.get_system(conn, "k")[0] == {"v": 1}
