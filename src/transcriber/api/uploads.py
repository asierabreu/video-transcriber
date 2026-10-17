"""Streaming multipart upload straight to disk, enforcing the size limit while streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import ClientDisconnect, Request

from transcriber.errors import ApiError

MAX_FIELD_BYTES = 10_000
MAX_FIELDS = 20


@dataclass
class UploadResult:
    """Parsed form fields and the stored file."""

    fields: dict[str, str] = field(default_factory=dict)
    filename: str | None = None
    size_bytes: int = 0


class _Parser:
    def __init__(self, dest: Path, max_bytes: int) -> None:
        self.dest = dest
        self.max_bytes = max_bytes
        self.result = UploadResult()
        self._header_field = b""
        self._header_value = b""
        self._headers: dict[bytes, bytes] = {}
        self._name: str | None = None
        self._is_file = False
        self._buffer = bytearray()
        self._fh: IO[bytes] | None = None
        self.got_file = False

    def on_part_begin(self) -> None:
        self._headers, self._name, self._is_file = {}, None, False
        self._buffer = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        self._headers[self._header_field.strip().lower()] = self._header_value.strip()
        self._header_field, self._header_value = b"", b""

    def on_headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = params.get(b"name", b"").decode("utf-8", "replace")
        filename = params.get(b"filename")
        self._name = name
        if name == "file" and filename is not None:
            if self.got_file:
                raise ApiError(400, "invalid_options", "Only one file may be uploaded.")
            self.got_file = True
            self._is_file = True
            self.result.filename = filename.decode("utf-8", "replace")
            self._fh = open(self.dest, "wb")  # noqa: SIM115 - closed in on_part_end/close
        elif len(self.result.fields) >= MAX_FIELDS:
            raise ApiError(400, "invalid_options", "Too many form fields.")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._is_file:
            assert self._fh is not None
            self.result.size_bytes += len(chunk)
            if self.result.size_bytes > self.max_bytes:
                raise ApiError(413, "file_too_large",
                               f"File exceeds the limit of {self.max_bytes // 2**20} MB.",
                               {"max_upload_mb": self.max_bytes // 2**20})
            self._fh.write(chunk)
        else:
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_FIELD_BYTES:
                raise ApiError(400, "invalid_options", f"Field {self._name!r} is too large.")

    def on_part_end(self) -> None:
        if self._is_file:
            self.close()
        elif self._name:
            self.result.fields[self._name] = self._buffer.decode("utf-8", "replace")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


async def stream_upload(request: Request, dest: Path, max_bytes: int) -> UploadResult:
    """Parse a multipart/form-data body, writing the `file` part to `dest` in chunks.

    Raises ApiError on malformed bodies, oversize files, or client disconnects. On error the
    partial file is removed.
    """
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise ApiError(400, "invalid_options", "Request must be multipart/form-data.")
    state = _Parser(dest, max_bytes)
    parser = MultipartParser(boundary, {
        "on_part_begin": state.on_part_begin,
        "on_part_data": state.on_part_data,
        "on_part_end": state.on_part_end,
        "on_header_field": state.on_header_field,
        "on_header_value": state.on_header_value,
        "on_header_end": state.on_header_end,
        "on_headers_finished": state.on_headers_finished,
    })
    try:
        async for chunk in request.stream():
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except ApiError:
        state.close()
        dest.unlink(missing_ok=True)
        raise
    except ClientDisconnect:
        state.close()
        dest.unlink(missing_ok=True)
        raise ApiError(400, "invalid_options", "Client disconnected during upload.") from None
    except Exception as exc:  # noqa: BLE001 - python-multipart raises various parse errors
        state.close()
        dest.unlink(missing_ok=True)
        raise ApiError(400, "invalid_options", f"Malformed multipart body: {exc}") from exc
    state.close()
    if not state.got_file:
        raise ApiError(400, "invalid_options", "A 'file' field with the video is required.")
    return state.result
