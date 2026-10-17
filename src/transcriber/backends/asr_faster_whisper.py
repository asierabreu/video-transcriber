"""faster-whisper ASR backend (faster-whisper 1.2.1, CTranslate2 4.x)."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from transcriber.backends.base import AsrResult, AsrWord, ProgressFn
from transcriber.errors import ModelLoadError

log = logging.getLogger(__name__)


def resolve_asr_model(model: str, models_dir: Path) -> str:
    """Map a model name to the local directory written by `make models`, if present."""
    if Path(model).is_dir():
        return model
    local = models_dir / "whisper" / model
    if local.is_dir():
        return str(local)
    return model


def asr_model_available(model: str, models_dir: Path) -> bool:
    """True if the model can be loaded without network access."""
    return Path(resolve_asr_model(model, models_dir)).is_dir()


def free_torch_memory() -> None:
    """Return cached GPU memory to the driver after a model is released."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()


class FasterWhisperAsr:
    """Transcribes English audio with word timestamps and VAD."""

    def __init__(
        self, model: str, models_dir: Path, device: str, compute_type: str, batch_size: int,
        beam_size: int, cpu_threads: int,
    ) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.name = model
        self.batch_size = batch_size
        self.beam_size = beam_size
        model_ref = resolve_asr_model(model, models_dir)
        try:
            self._model = WhisperModel(
                model_ref,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads if device == "cpu" else 0,
                download_root=str(models_dir / "whisper"),
                local_files_only=True,
            )
        except (OSError, ValueError) as exc:
            # Missing files surface as OSError/ValueError; CUDA/OOM problems as RuntimeError,
            # which the caller treats as a resource problem and handles via fallback.
            raise ModelLoadError(
                f"cannot load ASR model {model!r} from {model_ref}: {exc}"
            ) from exc
        self._batched = BatchedInferencePipeline(model=self._model) if batch_size > 0 else None
        log.info("ASR model loaded", extra={"model": model, "device": device,
                                            "compute_type": compute_type,
                                            "batch_size": batch_size})

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> AsrResult:
        """Run transcription; consumes the segment generator to completion."""
        if self._batched is not None:
            segments, info = self._batched.transcribe(
                str(audio_path), language="en", beam_size=self.beam_size,
                word_timestamps=True, vad_filter=True, batch_size=self.batch_size,
            )
        else:
            segments, info = self._model.transcribe(
                str(audio_path), language="en", beam_size=self.beam_size,
                word_timestamps=True, vad_filter=True,
            )
        duration = float(info.duration) or 1.0
        words: list[AsrWord] = []
        seg_out = []
        for seg in segments:
            seg_words = []
            for w in seg.words or []:
                text = w.word.strip()
                if not text:
                    continue
                item: AsrWord = {"w": text, "start": round(float(w.start), 3),
                                 "end": round(float(w.end), 3),
                                 "p": round(float(w.probability), 4)}
                seg_words.append(item)
            words.extend(seg_words)
            seg_out.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                            "text": seg.text.strip(), "n_words": len(seg_words)})
            progress(min(1.0, float(seg.end) / duration))
        words.sort(key=lambda w: (w["start"], w["end"]))
        return {"language": info.language or "en", "duration_s": round(float(info.duration), 3),
                "words": words, "segments": seg_out}

    def close(self) -> None:
        """Drop the model and free memory."""
        self._batched = None
        del self._model
        free_torch_memory()
