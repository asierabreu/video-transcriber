"""API application factory and the `transcriber-api` entry point."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from transcriber import __version__
from transcriber.api.routes import health, jobs, metrics, system
from transcriber.config import Settings, get_settings
from transcriber.errors import ApiError
from transcriber.logging import request_id_var, setup_logging
from transcriber.store.db import open_db

log = logging.getLogger("transcriber.api")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}


class RequestIdMiddleware:
    """Pure ASGI middleware (does not buffer bodies, so uploads still stream)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


def _error(status: int, code: str, message: str, details: dict[str, Any] | None = None,
           ) -> JSONResponse:
    return JSONResponse(ApiError(status, code, message, details).to_body(), status_code=status)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.to_body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return _error(400, "invalid_options", "Invalid request parameters.", {"errors": errors})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return _error(500, "internal_error", "An unexpected error occurred.",
                      {"request_id": request_id_var.get()})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Fails fast if API_KEY is missing or too short."""
    settings = settings or get_settings()
    settings.validate_for_api()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    open_db(settings.db_path).close()

    app = FastAPI(
        title="Video Transcriber",
        version=__version__,
        description="Self-hosted transcription with speaker labels and local LLM cleanup.",
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    _install_error_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    for module in (jobs, system, health, metrics):
        app.include_router(module.router)
    return app


def create_app_from_env() -> FastAPI:
    """Factory for `uvicorn --factory` (used by Procfile.dev): settings and logging from env."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    return create_app(settings)


def run() -> None:
    """Entry point: run Uvicorn with the configured host and port."""
    settings = get_settings()
    uvicorn.run(
        create_app_from_env(), host=settings.api_host, port=settings.api_port,
        # Trusted proxy IPs come from uvicorn's FORWARDED_ALLOW_IPS env var (default 127.0.0.1).
        log_config=None, proxy_headers=True, timeout_keep_alive=30,
    )


if __name__ == "__main__":
    run()
