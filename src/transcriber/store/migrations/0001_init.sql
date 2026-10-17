CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    stage            TEXT,
    progress         REAL NOT NULL DEFAULT 0,
    options          TEXT NOT NULL,
    media            TEXT NOT NULL,
    result           TEXT,
    warnings         TEXT NOT NULL DEFAULT '[]',
    error            TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    not_before       TEXT,
    worker_id        TEXT,
    heartbeat_at     TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS system (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
