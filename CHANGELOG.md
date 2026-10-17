# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-17

### Added
- HTTP API for submitting videos (MP4, MKV, MOV, AVI), tracking jobs, and downloading
  transcripts as JSON, TXT, SRT and VTT plus a Markdown summary.
- Worker pipeline: FFmpeg audio extraction, faster-whisper transcription with word timestamps,
  pyannote community-1 speaker diarization, word-level speaker alignment, Ollama-based cleanup
  with an edit guard, and map-reduce summaries.
- Automatic hardware tiers for NVIDIA GPUs, Apple Silicon and CPU-only hosts, with a per-job
  fallback chain on GPU memory errors.
- SQLite job queue with retries, resume from the last completed stage, cancellation and crash
  recovery.
- Health, readiness and Prometheus metrics endpoints.
- Docker images (CPU and CUDA), docker compose profiles, model download/export tooling.
