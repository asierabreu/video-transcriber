"""Detect the available accelerator once at worker startup (§5.1)."""

from __future__ import annotations

import logging
import os
import platform
import sys

from transcriber.config import Settings
from transcriber.hardware.tiers import HardwareProfile, select_tier

log = logging.getLogger(__name__)


def _cuda_info() -> tuple[bool, str | None, float | None, str | None]:
    """Return (usable, gpu_name, vram_gb, reason_if_not_usable)."""
    try:
        import torch
    except ImportError:
        return False, None, None, "torch is not installed"
    if not torch.cuda.is_available():
        return False, None, None, "torch.cuda.is_available() is False"
    try:
        import ctranslate2

        ct2_devices = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001 - any import/runtime failure means "no CUDA for ASR"
        return False, None, None, f"ctranslate2 CUDA check failed: {exc}"
    if ct2_devices < 1:
        return False, None, None, "ctranslate2 sees no CUDA devices (check cuDNN 9 / CUDA 12)"
    props = torch.cuda.get_device_properties(0)
    return True, props.name, round(props.total_memory / 2**30, 1), None


def _mps_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


def detect_hardware(settings: Settings) -> HardwareProfile:
    """Build the hardware profile, honouring the DEVICE override."""
    cores = os.cpu_count() or 1
    profile = HardwareProfile(cpu_threads=min(cores, settings.cpu_threads or cores))

    if settings.device != "auto":
        profile.device = settings.device
        profile.notes.append(f"DEVICE override: {settings.device}")
        if settings.device == "cuda":
            ok, name, vram, reason = _cuda_info()
            if ok:
                profile.gpu_name, profile.vram_gb = name, vram
            else:
                profile.notes.append(f"CUDA forced but not usable: {reason}")
                log.warning("DEVICE=cuda but CUDA is not usable: %s", reason)
    else:
        ok, name, vram, reason = _cuda_info()
        if ok:
            profile.device, profile.gpu_name, profile.vram_gb = "cuda", name, vram
        else:
            if reason and reason != "torch.cuda.is_available() is False":
                log.warning("CUDA not used: %s", reason)
                profile.notes.append(f"CUDA not used: {reason}")
            profile.device = "apple" if _mps_available() else "cpu"

    profile.tier = select_tier(profile.device, profile.vram_gb)
    log.info("hardware detected", extra={"hardware": profile.to_dict()})
    return profile
