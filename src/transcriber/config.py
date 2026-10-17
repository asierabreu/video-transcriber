"""Service configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Maps the user-facing container names to tokens that appear in ffprobe's `format_name`.
CONTAINER_ALIASES: dict[str, str] = {"mp4": "mp4", "mov": "mov", "mkv": "matroska", "avi": "avi"}


class Settings(BaseSettings):
    """All runtime settings. See docs/ARCHITECTURE.md §9 for descriptions."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        env_ignore_empty=True,  # `ASR_MODEL=` in .env means "use the tier default"
    )

    # API
    api_key: str = Field(default="", repr=False)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    docs_enabled: bool = False

    # Storage and limits
    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    max_upload_mb: int = Field(default=8192, ge=1)
    max_duration_s: int = Field(default=14400, ge=1)
    allowed_containers: str = "mp4,mkv,mov,avi"
    max_queued_jobs: int = Field(default=50, ge=1)
    min_free_disk_gb: float = Field(default=20, ge=0)
    keep_media: bool = False
    retention_days: int = Field(default=2, ge=0)

    # Hardware and ASR
    device: Literal["auto", "cuda", "apple", "cpu"] = "auto"
    asr_model: str | None = None
    asr_compute_type: Literal["float16", "int8_float16", "int8", "float32"] | None = None
    asr_batch_size: int | None = Field(default=None, ge=0)
    asr_beam_size: int = Field(default=5, ge=1)
    cpu_threads: int | None = Field(default=None, ge=1)

    # Diarization
    diarization_model: Path | None = None
    diarization_device: Literal["cuda", "mps", "cpu"] | None = None

    # LLM
    ollama_url: str = "http://ollama:11434"
    llm_model: str | None = None
    llm_num_ctx: int | None = Field(default=None, ge=2048)
    llm_timeout_s: float = Field(default=300, gt=0)
    llm_max_edit_ratio: float = Field(default=0.25, ge=0, le=1)
    llm_keep_alive: str = "5m"

    # Worker
    worker_concurrency: int = Field(default=1, ge=1, le=1)
    worker_poll_interval_s: float = Field(default=2, gt=0)
    job_max_attempts: int = Field(default=3, ge=1)

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Third-party library switches (exported to os.environ by `apply_library_env`).
    hf_hub_offline: bool = True
    pyannote_metrics_enabled: bool = False

    @field_validator("allowed_containers")
    @classmethod
    def _check_containers(cls, value: str) -> str:
        names = [c.strip().lower() for c in value.split(",") if c.strip()]
        unknown = [c for c in names if c not in CONTAINER_ALIASES]
        if not names or unknown:
            raise ValueError(
                f"ALLOWED_CONTAINERS must be a subset of {sorted(CONTAINER_ALIASES)}; got {value!r}"
            )
        return ",".join(names)

    @property
    def allowed_container_list(self) -> list[str]:
        """Allowed containers as a list of user-facing names."""
        return self.allowed_containers.split(",")

    @property
    def db_path(self) -> Path:
        """Location of the SQLite database."""
        return self.data_dir / "jobs.db"

    @property
    def jobs_dir(self) -> Path:
        """Root directory that holds one sub-directory per job."""
        return self.data_dir / "jobs"

    @property
    def diarization_model_path(self) -> Path:
        """Local pyannote pipeline directory."""
        if self.diarization_model is not None:
            return self.diarization_model
        return self.models_dir / "pyannote" / "speaker-diarization-community-1"

    @property
    def max_upload_bytes(self) -> int:
        """Upload limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    def validate_for_api(self) -> None:
        """Raise ValueError when the API cannot start safely."""
        if len(self.api_key) < 32:
            raise ValueError("API_KEY must be set and at least 32 characters long")

    def apply_library_env(self) -> None:
        """Export settings that third-party libraries read from the environment.

        Must run before faster-whisper, huggingface_hub or pyannote are imported.
        """
        os.environ["HF_HUB_OFFLINE"] = "1" if self.hf_hub_offline else "0"
        os.environ["PYANNOTE_METRICS_ENABLED"] = "1" if self.pyannote_metrics_enabled else "0"
        os.environ.setdefault("HF_HOME", str(self.models_dir / "hf"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
