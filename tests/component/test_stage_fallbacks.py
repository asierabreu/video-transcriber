import pytest

from tests.conftest import FakeFactory
from transcriber.config import Settings
from transcriber.errors import ModelLoadError, ResourceExhaustedError
from transcriber.hardware.tiers import HardwareProfile, resolve_effective
from transcriber.pipeline import diarize, transcribe
from transcriber.pipeline.stages import JobContext
from transcriber.store.files import JobFiles

JOB_ID = "01JAB3K9ZQ4W1X8T6M2N5P7R0S"


def context(tmp_path, factory, device="cuda"):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    tier = "cuda-large" if device == "cuda" else "cpu"
    eff = resolve_effective(settings, HardwareProfile(device=device, tier=tier))
    files = JobFiles(settings.jobs_dir, JOB_ID)
    files.dir.mkdir(parents=True)
    files.audio.write_bytes(b"RIFF")
    return JobContext(job_id=JOB_ID, files=files, options={}, media={}, eff=eff,
                      settings=settings, factory=factory, attempts=1)


def test_transcribe_falls_back_on_runtime_errors(tmp_path):
    factory = FakeFactory(asr_errors=[RuntimeError("CUDA out of memory")],
                          models={"large-v3", "distil-large-v3"})
    ctx = context(tmp_path, factory)
    transcribe.run(ctx)
    assert [a.compute_type for a in factory.asr_attempts] == ["float16", "int8_float16"]
    assert ctx.files.read_json("asr.json")["model"] == "large-v3"
    assert any("int8_float16" in w for w in ctx.warnings)


def test_transcribe_exhausts_chain(tmp_path):
    errors = [RuntimeError("oom")] * 4
    ctx = context(tmp_path, FakeFactory(asr_errors=errors, models={"large-v3",
                                                                   "distil-large-v3"}))
    with pytest.raises(ResourceExhaustedError):
        transcribe.run(ctx)


def test_missing_model_is_permanent(tmp_path):
    ctx = context(tmp_path, FakeFactory(asr_errors=[ModelLoadError("missing")]))
    with pytest.raises(ModelLoadError):
        transcribe.run(ctx)


def test_diarization_falls_back_to_cpu(tmp_path):
    factory = FakeFactory(diar_errors=[RuntimeError("CUDA error")])
    ctx = context(tmp_path, factory)
    ctx.files.write_json("asr.json", {"words": [{"w": "a", "start": 0, "end": 1, "p": 1}]})
    diarize.run(ctx)
    assert factory.diar_devices == ["cuda", "cpu"]
    assert ctx.files.read_json("diarization.json")["device"] == "cpu"


def test_diarization_skipped_without_words(tmp_path):
    factory = FakeFactory()
    ctx = context(tmp_path, factory, device="cpu")
    ctx.files.write_json("asr.json", {"words": []})
    diarize.run(ctx)
    assert factory.diar_devices == [] and ctx.files.read_json("diarization.json")["turns"] == []
