# Video Transcriber

Self-hosted video transcription with speaker labels. Upload a video over HTTP and get back a
transcript as JSON, plain text, SRT and VTT, plus a Markdown summary. Everything runs on your
own machine: audio extraction with FFmpeg, speech recognition with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), speaker diarization with
[pyannote.audio](https://github.com/pyannote/pyannote-audio), and cleanup and summaries with a
local LLM served by [Ollama](https://ollama.com). Nothing leaves the host once the models are
downloaded.

The design is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features

- MP4, MKV, MOV and AVI input, up to 4 hours per file (English speech).
- Word-level speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) grouped into speaker turns.
- LLM cleanup of punctuation, casing and obvious mis-recognitions, with a guard that keeps the
  original text when the model changes more than 25% of the words. The raw ASR text is always
  kept alongside.
- Map-reduce summaries with fixed sections: Overview, Key points, Decisions, Action items and
  Topics.
- Automatic hardware detection: NVIDIA GPU (three tiers by VRAM), Apple Silicon or CPU-only.
  Falls back to smaller models or the CPU on out-of-memory errors.
- A durable SQLite job queue with retries, resume from the last completed stage, cancellation,
  and 2-day retention by default.
- Shared API key auth, health and readiness probes, and Prometheus metrics.

## Hardware requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04/24.04 with Docker Engine + Compose v2, or macOS 14+ on Apple Silicon (development) | |
| RAM | 16 GB | 32 GB |
| Disk | 100 GB free (models are about 10 GB) | SSD |
| GPU | optional | NVIDIA with 8–16 GB+ VRAM, driver R570+ (CUDA 12.8), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

The worker picks a tier at startup and logs it. `GET /v1/system` reports it too.

| Tier | When | ASR model | LLM | 1-hour file (estimate) |
|---|---|---|---|---|
| `cuda-large` | VRAM ≥ 16 GB | large-v3, float16 | qwen3:8b | 4–7 min |
| `cuda-medium` | 8–16 GB | large-v3-turbo, int8_float16 | qwen3:8b | 6–10 min |
| `cuda-small` | < 8 GB | distil-large-v3, int8_float16 | qwen3:4b | — |
| `apple` | Apple Silicon (native only) | distil-large-v3, int8 on CPU | qwen3:8b | 20–35 min |
| `cpu` | otherwise | distil-large-v3, int8 | qwen3:4b | 60–120 min |

Any default can be overridden in `.env` (`ASR_MODEL`, `LLM_MODEL`, `DEVICE`, …).

## Quickstart with Docker (Ubuntu server)

```bash
git clone <this repo> video-transcriber && cd video-transcriber
cp .env.example .env
make api-key                       # paste the output into API_KEY= in .env
```

**1. Get a Hugging Face token for the diarization model.** This is a one-time step. Accept the
conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>, create a read
token at <https://huggingface.co/settings/tokens>, and set `HF_TOKEN=` in `.env`. The token is
only used by the download step, not by the running service.

**2. Download the models.** This is also one-time and needs internet access. It takes 10–30
minutes.

```bash
make models                 # every tier (about 10 GB)
make models TIER=cuda-medium   # or only what your hardware tier needs
```

With the default `TIER=all`, this downloads:

- ASR: `large-v3`, `large-v3-turbo`, and `distil-large-v3` for faster-whisper.
- Diarization: `pyannote/speaker-diarization-community-1` from Hugging Face.
- Ollama: `qwen3:8b` and `qwen3:4b`.

With a specific tier, only that tier's ASR and LLM models are downloaded, plus the
diarization model. For example, `TIER=cuda-medium` downloads `large-v3-turbo` and
`qwen3:8b`.

`TIER` accepts these values:

| Value | ASR model | Ollama model | Intended hardware |
|---|---|---|---|
| `all` | `large-v3`, `large-v3-turbo`, `distil-large-v3` | `qwen3:8b`, `qwen3:4b` | Download every tier |
| `cuda-large` | `large-v3` | `qwen3:8b` | NVIDIA GPU with ample VRAM |
| `cuda-medium` | `large-v3-turbo` | `qwen3:8b` | NVIDIA GPU with medium VRAM |
| `cuda-small` | `distil-large-v3` | `qwen3:4b` | NVIDIA GPU with limited VRAM |
| `apple` | `distil-large-v3` | `qwen3:8b` | Apple Silicon |
| `cpu` | `distil-large-v3` | `qwen3:4b` | CPU-only hosts |

Every tier also downloads `pyannote/speaker-diarization-community-1` for speaker
diarization. The `TIER` value can be supplied as an environment variable or Make
argument, for example `TIER=cpu make models`. The downloader script has additional
advanced flags (`--skip-asr`, `--skip-diarization`, `--skip-llm`, `--asr-model`, and
`--llm-model`), but those are not exposed by the `make models` target.

**3. Start the service.**

```bash
make up-gpu     # NVIDIA host
make up-cpu     # CPU-only host
```

**4. Check that it is ready.**

```bash
curl -s localhost:8000/readyz | python3 -m json.tool
```

The API listens on port 8000 (set `API_PUBLISH` to change it). Put it behind a TLS reverse proxy
before exposing it beyond a trusted network (see [Operations](docs/OPERATIONS.md#tls)).

**5. Stop the service.**

```bash
make stop    # stop CPU or GPU containers and preserve them for the next start
make down    # stop and remove the Compose containers and network
```

The worker receives a graceful stop signal and finishes its current stage before exiting. An
interrupted job resumes from its last completed stage when the service starts again. For native
development with `make dev`, press `Ctrl-C` to stop the API and worker.

## Local development (macOS, Apple Silicon)

Docker on macOS cannot use the Apple GPU, so run natively:

```bash
brew install ffmpeg uv
# install the Ollama app from https://ollama.com/download and start it
cp .env.example .env
# edit .env: API_KEY=<make api-key>, HF_TOKEN=<token>,
#            DATA_DIR=./.data, MODELS_DIR=./.models, OLLAMA_URL=http://localhost:11434
make install                 # uv provisions Python 3.11 and installs everything
make models-local TIER=apple
make dev                     # API on http://127.0.0.1:8000 plus the worker, with auto-reload
```

The `apple` tier activates automatically. Diarization runs on the Apple GPU (MPS), Ollama
uses Metal, and transcription runs on the CPU, because CTranslate2 has no Metal backend. The
same steps work on Linux without Docker; use `TIER=cpu` or a CUDA tier.

## Using the API

Every `/v1` endpoint needs the `X-API-Key` header. The examples assume:

```bash
export API=http://localhost:8000 KEY=<your API_KEY>
```

**Submit a video** (multipart; all fields except `file` are optional):

```bash
curl -s -X POST "$API/v1/jobs" -H "X-API-Key: $KEY" \
  -F file=@meeting.mp4 -F diarization=true -F num_speakers=3 -F cleanup=true -F summary=true -F title="Weekly sync"
# {"id":"01JAB3K9ZQ4W1X8T6M2N5P7R0S","status":"queued","queue_position":1,...}
export JOB=01JAB3K9ZQ4W1X8T6M2N5P7R0S
```

Use `min_speakers` and `max_speakers` instead of `num_speakers` when the speaker count is only
roughly known.

The default is the fast transcript-only pipeline. Enable optional processing explicitly:

```bash
# Fast default: transcript and subtitle files, no speaker diarization or LLM calls.
./scripts/transcriber.py submit meeting.mp4

# Add speaker labels, without using the LLM.
./scripts/transcriber.py submit meeting.mp4 --diarization --num-speakers 3

# Correct and normalize transcript text, without speaker diarization or a summary.
./scripts/transcriber.py submit meeting.mp4 --cleanup

# Generate a summary; this also prepares the transcript for the summarizer.
./scripts/transcriber.py submit meeting.mp4 --summary

# Enable every optional stage.
./scripts/transcriber.py submit meeting.mp4 --diarization --cleanup --summary \
  --num-speakers 3 --title "Weekly sync"
```

The equivalent full-processing upload with curl is:

```bash
curl -s -X POST "$API/v1/jobs" -H "X-API-Key: $KEY" \
  -F file=@meeting.mp4 -F diarization=true -F num_speakers=3 \
  -F cleanup=true -F summary=true -F title="Weekly sync"
```

**Check progress:**

```bash
# Directly using API
curl -s "$API/v1/jobs/$JOB" -H "X-API-Key: $KEY" | python3 -m json.tool

# Using watch_job wrapper : watch status, stages, and overall progress until the job finishes.
./scripts/watch_job.py "$JOB"

# Wrapper with adjustable polling interval:
./scripts/watch_job.py --interval 5 "$JOB"
```

**List jobs** (newest first, paginated):

```bash
curl -s "$API/v1/jobs?status=succeeded&limit=20" -H "X-API-Key: $KEY"
curl -s "$API/v1/jobs?cursor=<next_cursor from the previous page>" -H "X-API-Key: $KEY"
```

**Download results** once `status` is `succeeded`:

```bash
curl -s "$API/v1/jobs/$JOB/transcript" -H "X-API-Key: $KEY" -o transcript.json
curl -s "$API/v1/jobs/$JOB/transcript?format=txt" -H "X-API-Key: $KEY" -o transcript.txt
curl -s "$API/v1/jobs/$JOB/transcript?format=srt" -H "X-API-Key: $KEY" -o transcript.srt
curl -s "$API/v1/jobs/$JOB/transcript?format=vtt" -H "X-API-Key: $KEY" -o transcript.vtt
curl -s "$API/v1/jobs/$JOB/summary" -H "X-API-Key: $KEY" -o summary.md
```

**Cancel or delete a job.** A queued or running job is cancelled. A finished job is deleted.
Either way, its files are removed.

```bash
curl -s -X DELETE "$API/v1/jobs/$JOB" -H "X-API-Key: $KEY" -w "%{http_code}\n"   # 204
```

**System, health and metrics:**

```bash
curl -s "$API/v1/system" -H "X-API-Key: $KEY"   # hardware tier, models, worker, queue
curl -s "$API/healthz"                           # liveness (no auth)
curl -s "$API/readyz"                            # readiness: DB, disk, worker; reports Ollama
curl -s "$API/metrics"                           # Prometheus text format
```

### Transcriber script

The executable transcriber wraps the API workflows above. Install its Python dependencies once
with `python3 -m pip install -e .`. It uses the `API`, `KEY`, and `JOB` environment variables
shown earlier; run `./scripts/transcriber.py --help` for all options.

```bash
./scripts/transcriber.py submit meeting.mp4 --diarization --cleanup --summary \
  --num-speakers 3 --title "Weekly sync"
./scripts/transcriber.py watch "$JOB" --interval 5
./scripts/transcriber.py job "$JOB"
./scripts/transcriber.py list --status succeeded --limit 20
./scripts/transcriber.py download "$JOB" --format json --output transcript.json
./scripts/transcriber.py download "$JOB" --format txt --output transcript.txt
./scripts/transcriber.py download "$JOB" --format summary --output summary.md
./scripts/transcriber.py delete "$JOB"
./scripts/transcriber.py system
./scripts/transcriber.py health
./scripts/transcriber.py ready
./scripts/transcriber.py metrics
```

## Pipeline stages

Jobs run through these stages in order. The API exposes the active stage and overall progress
while a job is running. The default fast configuration runs only extracting, transcribing, and
rendering. This avoids the diarization and LLM stages for the shortest processing time.

1. **Extracting**: decodes the uploaded video and creates a 16 kHz mono WAV for analysis.
2. **Transcribing**: runs Whisper ASR to produce segments, words, timestamps, and raw text.
3. **Diarizing**: optionally identifies speaker turns and assigns speaker labels to the transcript.
4. **Aligning**: reconciles ASR words with diarization turns and builds the aligned transcript.
5. **Cleaning**: optionally uses the configured LLM to correct and normalize transcript text.
6. **Summarizing**: optionally uses the LLM to generate the structured Markdown summary.
7. **Rendering**: writes the final JSON, TXT, SRT, and VTT transcript files.

Enable additional work per job with `diarization=true`, `cleanup=true`, and `summary=true`.
These options can be combined independently. For example, `diarization=true` adds speaker
labels without enabling either LLM stage; `cleanup=true` or `summary=true` also adds the
alignment and cleanup prerequisites automatically.

Errors always look like
`{"error": {"code": "unsupported_media", "message": "...", "details": {...}}}`. The full
reference is in [docs/API.md](docs/API.md). Set `DOCS_ENABLED=true` to serve interactive docs at
`/docs`.

## Configuration

All settings are environment variables, read from `.env`. [.env.example](.env.example) lists
every one of them, and [docs/ARCHITECTURE.md §9](docs/ARCHITECTURE.md#9-configuration)
describes them. These are the ones you are most likely to change:

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | — | Required, 32+ characters |
| `LLM_MODEL` | tier default (`qwen3:8b`) | Any Ollama tag. Run `docker compose exec ollama ollama pull <tag>` first |
| `RETENTION_DAYS` | `2` | Finished jobs and their files are deleted after this many days |
| `KEEP_MEDIA` | `false` | Keep the uploaded video and WAV after success |
| `MAX_UPLOAD_MB` / `MAX_DURATION_S` | `8192` / `14400` | Upload limits |
| `DEVICE` | `auto` | Force `cuda`, `apple` or `cpu` |

## Tests

```bash
make test        # 90+ tests: unit, component, API and integration; needs ffmpeg, no models
make lint typecheck
```

### End-to-end test

`make test-e2e` runs the whole stack with real models on a 51-second, two-speaker clip. The
clip is built on first run from two public-domain LibriVox recordings, downloaded from
archive.org and cached in `tests/fixtures/e2e_cache/`. The test checks that at least two
speakers are found, that the word error rate on the LibriVox disclaimer stays under 25%, and
that every output format parses. It needs the diarization model in `MODELS_DIR`, downloads
`base.en` if it is missing, and runs the LLM steps only if Ollama has `qwen3:0.6b` (run
`ollama pull qwen3:0.6b`). The test skips itself when any of these is unavailable. You can
override the defaults with `E2E_CLIP_URLS`, `E2E_ASR_MODEL` and `E2E_LLM_MODEL`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Worker exits with `required model missing` | Run `make models TIER=<tier from the log>`, or `make models-local` natively. |
| `make models` stops with `HF_TOKEN is required` / 401 / 403 | Accept the model conditions on Hugging Face with the same account as the token, and set `HF_TOKEN` in `.env`. |
| `/readyz` returns 503 with `worker.ok=false` | The worker is not running or has crashed. Check with `make logs`. It also exits when models are missing. |
| Jobs succeed with warning `cleanup skipped: model ... is not pulled` | Run `docker compose exec ollama ollama pull <LLM_MODEL>`. |
| Jobs stay `queued` with a growing `attempts` value | Ollama is unreachable. Jobs retry with backoff, then finish without cleanup or summary. Check `docker compose ps ollama`. |
| GPU host runs on CPU (`tier: cpu` in `/v1/system`, note `ctranslate2 sees no CUDA devices`) | Install the NVIDIA Container Toolkit, use `make up-gpu`, and check `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`. |
| `out_of_resources` failures or fallback warnings | The GPU is too small for the tier. Set `ASR_MODEL=large-v3-turbo` or `distil-large-v3`, or a smaller `LLM_MODEL`. |
| `413 file_too_large` behind nginx | Raise `client_max_body_size` in the proxy to match `MAX_UPLOAD_MB`. |
| `415 unsupported_media` for a valid file | Only MP4, MKV, MOV and AVI are accepted. Remux with `ffmpeg -i in.webm -c copy out.mkv`. |
| Speakers merged or split | Pass `num_speakers`, or `min_speakers` and `max_speakers`, when submitting. |

More in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Licenses

The code is MIT licensed (see [LICENSE](LICENSE)). The default models are permissively licensed:
Whisper weights are MIT and Qwen3 is Apache 2.0. The pyannote community-1 diarization weights
are **CC-BY-4.0, which requires attribution**. Keep [NOTICE](NOTICE) with any distribution.
FFmpeg is called as a separate program, and the Docker images use the distribution's GPL build;
see NOTICE.
