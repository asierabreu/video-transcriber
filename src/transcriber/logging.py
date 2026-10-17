"""Structured logging with request/job context."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

try:  # python-json-logger >= 3.1
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # pragma: no cover - python-json-logger 2.x
    from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[no-redef,unused-ignore]

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
stage_var: ContextVar[str | None] = ContextVar("stage", default=None)


class _ContextFilter(logging.Filter):
    """Copy context variables onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for name, var in (("request_id", request_id_var), ("job_id", job_id_var),
                          ("stage", stage_var)):
            value = var.get()
            if value is not None and not hasattr(record, name):
                setattr(record, name, value)
        return True


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root logger to write to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if fmt == "json":
        handler.setFormatter(
            JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        )
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
