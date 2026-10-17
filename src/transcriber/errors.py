"""Typed errors shared by the API and the worker."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """An error returned to API clients in the standard error envelope (§7.3)."""

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def to_body(self) -> dict[str, Any]:
        """Return the JSON body for this error."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class MediaValidationError(Exception):
    """Uploaded media failed ffprobe validation."""

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.api_error = ApiError(status, code, message, details)


class PipelineError(Exception):
    """Base class for worker pipeline failures.

    `code` is the failed-job error code (§7.3); `transient` decides whether the job is retried.
    """

    code = "internal_error"
    transient = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class MediaError(PipelineError):
    """Corrupt or unreadable media. Permanent."""

    code = "extraction_failed"


class ModelLoadError(PipelineError):
    """A required model is missing or cannot be loaded. Permanent."""

    code = "model_unavailable"


class ResourceExhaustedError(PipelineError):
    """Out of GPU/CPU memory or disk after all fallbacks. Transient."""

    code = "out_of_resources"
    transient = True


class LLMError(PipelineError):
    """The LLM runtime is unreachable or misbehaving. Transient."""

    code = "llm_unavailable"
    transient = True


class JobCancelled(Exception):
    """Raised inside the pipeline when the job's cancel flag is set."""


class WorkerStopping(Exception):
    """Raised inside the pipeline when the worker is shutting down."""
