"""Ollama LLM backend over its local HTTP API (/api/chat, /api/tags)."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from transcriber.backends.base import LlmStatus
from transcriber.errors import LLMError

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalize_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaLlm:
    """Chat completions against a local Ollama server.

    Every request disables reasoning output (`think: false`) and uses temperature 0. Models
    that reject the `think` field are retried once without it, and the result is remembered.
    """

    def __init__(
        self, base_url: str, model: str, num_ctx: int, timeout_s: float, keep_alive: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._send_think = True
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    def status(self) -> LlmStatus:
        """Check reachability and whether the model has been pulled."""
        try:
            resp = self._client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return LlmStatus(reachable=False, model_present=False, detail=str(exc))
        names = {_normalize_tag(m.get("name", "")) for m in resp.json().get("models", [])}
        present = _normalize_tag(self.name) in names
        detail = "" if present else f"model {self.name!r} is not pulled in Ollama"
        return LlmStatus(reachable=True, model_present=present, detail=detail)

    def chat(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one non-streaming chat request and return the reply text."""
        options: dict[str, Any] = {"temperature": 0, "num_ctx": self.num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if json_schema is not None:
            body["format"] = json_schema
        if self._send_think:
            body["think"] = False
        try:
            resp = self._client.post("/api/chat", json=body)
            if resp.status_code == 400 and "think" in resp.text.lower() and self._send_think:
                log.info("model rejects the think option; retrying without it",
                         extra={"model": self.name})
                self._send_think = False
                body.pop("think", None)
                resp = self._client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama returned {exc.response.status_code}: "
                           f"{exc.response.text[:200]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        try:
            content = resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an unexpected response body") from exc
        return _THINK_BLOCK.sub("", content).strip()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
