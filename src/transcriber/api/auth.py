"""Shared API-key authentication (§10)."""

from __future__ import annotations

import hmac
import logging

from fastapi import Request

from transcriber.errors import ApiError

log = logging.getLogger(__name__)


def require_api_key(request: Request) -> None:
    """Reject the request unless `X-API-Key` matches API_KEY (constant-time comparison)."""
    expected: str = request.app.state.settings.api_key
    provided = request.headers.get("x-api-key", "")
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        client = request.client.host if request.client else "unknown"
        log.warning("authentication failed", extra={"client_ip": client,
                                                    "path": request.url.path})
        raise ApiError(401, "unauthorized", "Missing or invalid API key.")
