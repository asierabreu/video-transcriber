"""Hardware tiers, default models, overrides and the ASR fallback chain (§5.2–5.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from transcriber.config import Settings

# Ordered largest to smallest; used for the per-job fallback chain (§5.3).
ASR_MODEL_CHAIN = ["large-v3", "large-v3-turbo", "distil-large-v3"]


@dataclass(frozen=True)
class Tier:
    """Default model settings for one class of hardware."""

    name: str
    asr_model: str
    asr_compute_type: str
    asr_device: str
    asr_batch_size: int
    diarization_device: str
    llm_model: str
    llm_num_ctx: int


TIERS: dict[str, Tier] = {
    "cuda-large": Tier("cuda-large", "large-v3", "float16", "cuda", 16, "cuda", "qwen3:8b", 16384),
    "cuda-medium": Tier("cuda-medium", "large-v3-turbo", "int8_float16", "cuda", 8, "cuda",
                        "qwen3:8b", 8192),
    "cuda-small": Tier("cuda-small", "distil-large-v3", "int8_float16", "cuda", 4, "cuda",
                       "qwen3:4b", 8192),
    "apple": Tier("apple", "distil-large-v3", "int8", "cpu", 0, "mps", "qwen3:8b", 8192),
    "cpu": Tier("cpu", "distil-large-v3", "int8", "cpu", 0, "cpu", "qwen3:4b", 8192),
}


@dataclass
class HardwareProfile:
    """What the worker found at startup."""

    device: str = "cpu"
    tier: str = "cpu"
    gpu_name: str | None = None
    vram_gb: float | None = None
    cpu_threads: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveConfig:
    """Tier defaults with operator overrides applied; recorded in each job's meta.json."""

    tier: str
    asr_model: str
    asr_compute_type: str
    asr_device: str
    asr_batch_size: int
    asr_beam_size: int
    cpu_threads: int
    diarization_device: str
    llm_model: str
    llm_num_ctx: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_tier(device: str, vram_gb: float | None) -> str:
    """Pick a tier name from the detected device and GPU memory."""
    if device == "cuda":
        vram = vram_gb or 0.0
        if vram >= 16:
            return "cuda-large"
        if vram >= 8:
            return "cuda-medium"
        return "cuda-small"
    if device == "apple":
        return "apple"
    return "cpu"


def resolve_effective(settings: Settings, profile: HardwareProfile) -> EffectiveConfig:
    """Combine the tier defaults with any explicit overrides from settings (§5.4)."""
    tier = TIERS[profile.tier]
    compute = settings.asr_compute_type or tier.asr_compute_type
    asr_device = tier.asr_device
    # A CPU device cannot run float16 kernels in CTranslate2.
    if asr_device == "cpu" and compute in ("float16", "int8_float16"):
        compute = "int8"
    return EffectiveConfig(
        tier=tier.name,
        asr_model=settings.asr_model or tier.asr_model,
        asr_compute_type=compute,
        asr_device=asr_device,
        asr_batch_size=(settings.asr_batch_size if settings.asr_batch_size is not None
                        else tier.asr_batch_size),
        asr_beam_size=settings.asr_beam_size,
        cpu_threads=profile.cpu_threads,
        diarization_device=settings.diarization_device or tier.diarization_device,
        llm_model=settings.llm_model or tier.llm_model,
        llm_num_ctx=settings.llm_num_ctx or tier.llm_num_ctx,
    )


@dataclass(frozen=True)
class AsrAttempt:
    """One configuration to try when loading and running the ASR model."""

    model: str
    compute_type: str
    device: str
    batch_size: int
    reason: str | None = None


def asr_fallback_chain(eff: EffectiveConfig, available_models: set[str]) -> list[AsrAttempt]:
    """Ordered ASR configurations to try (§5.3).

    1. The effective configuration.
    2. On CUDA with float16: the same model with int8_float16.
    3. Each smaller model in ASR_MODEL_CHAIN that is available locally.
    4. CPU with int8, using the smallest model tried.
    """
    first = AsrAttempt(eff.asr_model, eff.asr_compute_type, eff.asr_device, eff.asr_batch_size)
    chain = [first]
    if first.device == "cuda" and first.compute_type == "float16":
        chain.append(replace(first, compute_type="int8_float16",
                             reason="retrying with int8_float16"))
    if first.device == "cuda" and eff.asr_model in ASR_MODEL_CHAIN:
        start = ASR_MODEL_CHAIN.index(eff.asr_model) + 1
        for model in ASR_MODEL_CHAIN[start:]:
            if model in available_models:
                chain.append(AsrAttempt(model, "int8_float16", "cuda", first.batch_size,
                                        reason=f"falling back to smaller model {model}"))
    if first.device != "cpu":
        last_model = chain[-1].model
        chain.append(AsrAttempt(last_model, "int8", "cpu", 0,
                                reason=f"falling back to CPU with {last_model}"))
    return chain
