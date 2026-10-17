"""pyannote.audio 4.x diarization backend (speaker-diarization-community-1)."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

from transcriber.backends.asr_faster_whisper import free_torch_memory
from transcriber.backends.base import SpeakerTurn
from transcriber.errors import MediaError, ModelLoadError

log = logging.getLogger(__name__)


def diarization_model_available(model_dir: Path) -> bool:
    """True if the local pipeline directory looks complete."""
    return (model_dir / "config.yaml").is_file()


def load_wav_tensor(path: Path) -> tuple[Any, int]:
    """Read a 16-bit PCM WAV into a (1, samples) float32 tensor without pyannote's decoder."""
    import numpy as np
    import torch

    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getsampwidth() != 2:
                raise MediaError(f"expected 16-bit PCM WAV, got {wf.getsampwidth() * 8}-bit")
            channels, rate = wf.getnchannels(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise MediaError(f"cannot read extracted audio: {exc}") from exc
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(samples).unsqueeze(0), rate


class PyannoteDiarizer:
    """Runs the community-1 pipeline and returns its exclusive diarization."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.name = "pyannote/speaker-diarization-community-1"
        if not diarization_model_available(model_dir):
            raise ModelLoadError(f"diarization model not found in {model_dir}; run `make models`")
        try:
            pipeline = Pipeline.from_pretrained(str(model_dir))
        except Exception as exc:  # noqa: BLE001 - pyannote raises many types on bad checkpoints
            raise ModelLoadError(f"cannot load diarization pipeline: {exc}") from exc
        if pipeline is None:
            raise ModelLoadError(f"cannot load diarization pipeline from {model_dir}")
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        self.device = device
        log.info("diarization pipeline loaded", extra={"device": device})

    def diarize(
        self, audio_path: Path, num_speakers: int | None, min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerTurn]:
        """Diarize the whole file in memory."""
        waveform, rate = load_wav_tensor(audio_path)
        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers
        output = self._pipeline({"waveform": waveform, "sample_rate": rate}, **kwargs)
        # pyannote 4 returns an output object; the exclusive annotation has one speaker at a time.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        turns: list[SpeakerTurn] = [
            {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
             "speaker": str(label)}
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t["start"])
        return turns

    def close(self) -> None:
        """Drop the pipeline and free memory."""
        del self._pipeline
        free_torch_memory()
