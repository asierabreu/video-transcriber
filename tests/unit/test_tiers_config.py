import pytest
from pydantic import ValidationError

from transcriber.config import Settings
from transcriber.hardware.tiers import (
    HardwareProfile,
    asr_fallback_chain,
    resolve_effective,
    select_tier,
)


def make_settings(**kw):
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize("device,vram,tier", [
    ("cuda", 24, "cuda-large"), ("cuda", 16, "cuda-large"), ("cuda", 12, "cuda-medium"),
    ("cuda", 6, "cuda-small"), ("cuda", None, "cuda-small"), ("apple", None, "apple"),
    ("cpu", None, "cpu"),
])
def test_select_tier(device, vram, tier):
    assert select_tier(device, vram) == tier


def test_tier_defaults_and_overrides():
    eff = resolve_effective(make_settings(), HardwareProfile(device="apple", tier="apple"))
    assert (eff.asr_model, eff.asr_device, eff.diarization_device, eff.llm_model) == (
        "distil-large-v3", "cpu", "mps", "qwen3:8b")
    eff = resolve_effective(
        make_settings(asr_model="large-v3", llm_model="llama3.1:8b", llm_num_ctx=4096,
                      asr_batch_size=0, diarization_device="cpu"),
        HardwareProfile(device="cuda", tier="cuda-medium"))
    assert eff.asr_model == "large-v3" and eff.llm_model == "llama3.1:8b"
    assert eff.llm_num_ctx == 4096 and eff.asr_batch_size == 0
    assert eff.diarization_device == "cpu"


def test_cpu_never_gets_float16():
    eff = resolve_effective(make_settings(asr_compute_type="float16"),
                            HardwareProfile(device="cpu", tier="cpu"))
    assert eff.asr_compute_type == "int8"


def test_fallback_chain_cuda_large():
    eff = resolve_effective(make_settings(), HardwareProfile(device="cuda", tier="cuda-large"))
    chain = asr_fallback_chain(eff, {"large-v3", "distil-large-v3"})
    assert [(a.model, a.compute_type, a.device) for a in chain] == [
        ("large-v3", "float16", "cuda"),
        ("large-v3", "int8_float16", "cuda"),
        ("distil-large-v3", "int8_float16", "cuda"),
        ("distil-large-v3", "int8", "cpu"),
    ]


def test_fallback_chain_cpu_has_single_attempt():
    eff = resolve_effective(make_settings(), HardwareProfile(device="cpu", tier="cpu"))
    assert len(asr_fallback_chain(eff, {"distil-large-v3"})) == 1


def test_config_validation():
    with pytest.raises(ValueError):
        make_settings(api_key="short").validate_for_api()
    make_settings(api_key="x" * 32).validate_for_api()
    with pytest.raises(ValidationError):
        make_settings(allowed_containers="mp4,flv")
    assert make_settings(allowed_containers="MKV, mp4").allowed_container_list == ["mkv", "mp4"]
    assert make_settings().retention_days == 2
