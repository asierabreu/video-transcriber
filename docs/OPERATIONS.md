# Operations

## Services

| Service | Image | Notes |
|---|---|---|
| `api` | `transcriber:cpu` | Serves port 8000. Holds no ML models. |
| `worker-cpu` / `worker-gpu` | `transcriber:cpu` / `transcriber:gpu` | Exactly one runs, selected by compose profile. |
| `ollama` | `ollama/ollama:${OLLAMA_TAG}` | Reachable only on the compose network. |
| `models` | `transcriber:cpu` | One-off downloader (`tools` profile). |

Volumes are `transcriber_data` (jobs and `jobs.db`), `transcriber_models` (Whisper and
pyannote weights) and `transcriber_ollama` (LLM weights).

## TLS

The API speaks plain HTTP, and the shared key is visible on the wire without TLS. Terminate TLS
in a reverse proxy. A minimal Caddyfile:

```
transcribe.example.com {
    request_body {
        max_size 8GB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 30m
        }
    }
}
```

Publish the API only on localhost when a proxy runs on the same host: set
`API_PUBLISH=127.0.0.1:8000` in the environment for compose. To log real client IPs, set
`FORWARDED_ALLOW_IPS=<proxy ip>` in `.env`. Uvicorn reads it.

## Rotating the API key

Change `API_KEY` in `.env`, then run `docker compose up -d api`. Clients need the new key
immediately.

## Changing models

- **LLM:** run `docker compose exec ollama ollama pull mistral:7b`, set `LLM_MODEL=mistral:7b`,
  then restart the worker. A model that is not pulled is detected per job. Those jobs succeed
  without cleanup or summary and carry a warning.
- **ASR:** set `ASR_MODEL`, then run `make models` with `--asr-model <name>`
  (`docker compose --profile tools run --rm models --asr-model large-v3 --skip-diarization --skip-llm`).
  The worker refuses to start if the configured model is missing.

## Retention and cleanup

- Finished jobs older than `RETENTION_DAYS` (default 2) are deleted, files first, by a purge
  that runs when the worker starts and then every 24 hours.
- Source videos and extracted WAVs are deleted as soon as a job succeeds, unless
  `KEEP_MEDIA=true`.
- Temp files and orphaned job directories older than 24 hours are removed by the same purge.

## Backups

Back up `jobs.db` safely while running with
`docker compose exec api python -c "import sqlite3; s=sqlite3.connect('/data/jobs.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d)"`,
then copy `/data` out of the volume. Models can always be downloaded again, or kept with
`make export-models`.

## Air-gapped hosts

On a connected machine, run `make models` and then `make export-models`. Copy
`models-bundle.tar.gz` to the target host, then run `make import-models` and `make up-gpu` (or
`make up-cpu`). The running service needs no internet access. You can add
`internal: true` to a dedicated compose network to enforce this.

## Failure handling

- **Worker crash or restart:** jobs whose heartbeat is older than 5 minutes are re-queued. They
  resume from the last completed stage, so finished transcription work is not repeated.
- **Ollama down:** jobs retry with backoff (30 s, 60 s, 120 s). On the last attempt they succeed
  with the raw transcript and a warning.
- **GPU out of memory:** the worker steps down from float16 to int8_float16, then to a smaller
  model, then to the CPU. Each step is recorded in `warnings`.

## Monitoring

Scrape `/metrics`. Useful alerts:

- `transcribe_worker_up == 0` for 2 minutes.
- `transcribe_queue_depth > 10`.
- `increase(transcribe_jobs_total{status="failed"}[1h]) > 0`.
- `transcribe_realtime_factor > 1` on a GPU host, which suggests it fell back to the CPU.

## Upgrading

Pull the new code, then run `make up-gpu` or `make up-cpu`. Database migrations run
automatically at start (`PRAGMA user_version`). Queued jobs survive the restart.
