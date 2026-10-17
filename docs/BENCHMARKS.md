# Benchmarks

Record measured results here with `make bench FILE=<1-hour recording>`. The figures below are
the planning estimates from the architecture document. Replace each row with a measurement on
your own hardware.

| Tier | Hardware | ASR model | LLM | 1 h media: processing time | RTF | Source |
|---|---|---|---|---|---|---|
| cuda-large | ≥ 16 GB NVIDIA | large-v3 fp16, batch 16 | qwen3:8b | 4–7 min | 0.07–0.12 | estimate |
| cuda-medium | 8–16 GB NVIDIA | large-v3-turbo int8_float16, batch 8 | qwen3:8b | 6–10 min | 0.10–0.17 | estimate |
| apple | M-series | distil-large-v3 int8 (CPU) | qwen3:8b (Metal) | 20–35 min | 0.33–0.58 | estimate |
| cpu | 8+ cores | distil-large-v3 int8 | qwen3:4b | 60–120 min | 1.0–2.0 | estimate |

## How to measure

1. Start the service on the target host and make sure `/readyz` is 200.
2. Run `API_KEY=... make bench FILE=recording.mp4`. To measure ASR and diarization only, run
   `python scripts/bench.py recording.mp4 --no-llm`.
3. Copy the per-stage output into a new row. Include the CPU or GPU model, driver and CUDA
   versions, and the `/v1/system` `models` block.
