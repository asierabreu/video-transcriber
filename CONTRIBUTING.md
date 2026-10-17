# Contributing

Thanks for helping improve Video Transcriber.

## Setup

```bash
make install        # uv sync --extra ml --extra dev (Python 3.11 is provisioned by uv)
cp .env.example .env   # set API_KEY, DATA_DIR=./.data, MODELS_DIR=./.models, OLLAMA_URL=http://localhost:11434
```

You need `ffmpeg` on your PATH for the integration tests (`brew install ffmpeg` or
`apt-get install ffmpeg`).

## Checks

Run these before opening a pull request; CI runs the same commands.

```bash
make lint           # ruff
make typecheck      # mypy
make test           # unit, component, API and integration tests (no models needed)
make test-e2e       # optional: real models, see README "End-to-end test"
make licenses       # fails on GPL/AGPL/LGPL or non-commercial Python dependencies
```

## Guidelines

- Read `docs/ARCHITECTURE.md` first. Changes to the API contract, data model or job state
  machine need an update to that document in the same pull request.
- Keep pipeline stages pure where possible: logic that can be tested without models
  (alignment, chunking, formats, guards) lives outside the backend classes.
- New ML backends implement the protocols in `src/transcriber/backends/base.py` and are wired
  in through `DefaultBackendFactory` (`src/transcriber/worker/runner.py`).
- Only permissively licensed dependencies (MIT, BSD, Apache-2.0, ISC, PSF, CC-BY for weights).
- Never log transcript text or API keys.
- Add a line to `CHANGELOG.md` under "Unreleased".

## Commit and pull request style

Small, focused commits with imperative subjects ("Add VTT voice tags"). Describe user-visible
changes and any migration steps in the pull request.
