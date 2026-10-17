# Test fixtures

Nothing large is committed here.

- Unit, component and API tests generate media at test time with ffmpeg's `lavfi` sources
  (`tests/conftest.py::make_media`). Canned ASR, diarization and LLM outputs come from the fake
  backends in the same file.
- `scripts/gen_fixtures.sh` writes sample videos to `generated/` for manual curl testing.
- The end-to-end test downloads two public-domain LibriVox clips on first run and caches them in
  `e2e_cache/`.

Both `generated/` and `e2e_cache/` are git-ignored.
