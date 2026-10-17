import json

import httpx
import pytest

from transcriber.backends.llm_ollama import OllamaLlm
from transcriber.errors import LLMError


def make(handler, model="qwen3:8b"):
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaLlm("http://ollama", model, 8192, 5, "5m", client=client)


def test_chat_sends_expected_body_and_strips_think_blocks():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "<think>x</think> hello "}})

    llm = make(handler)
    assert llm.chat("sys", "user", json_schema={"type": "object"}, max_tokens=50) == "hello"
    assert seen["think"] is False and seen["stream"] is False
    assert seen["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 50}
    assert seen["format"] == {"type": "object"} and seen["keep_alive"] == "5m"


def test_retries_without_think_when_model_rejects_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return httpx.Response(200, json={"message": {"content": "ok"}})

    llm = make(handler, "llama3.1:8b")
    assert llm.chat("s", "u") == "ok"
    assert llm.chat("s", "u") == "ok"
    assert len(bodies) == 3 and "think" not in bodies[2]


def test_http_errors_become_llm_errors():
    llm = make(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        llm.chat("s", "u")

    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        make(refuse).chat("s", "u")

    with pytest.raises(LLMError):
        make(lambda r: httpx.Response(200, text="not json")).chat("s", "u")


def test_status_checks_model_presence():
    tags = {"models": [{"name": "qwen3:8b"}, {"name": "mistral:latest"}]}
    assert make(lambda r: httpx.Response(200, json=tags)).status().usable
    assert make(lambda r: httpx.Response(200, json=tags), "mistral").status().usable
    missing = make(lambda r: httpx.Response(200, json=tags), "qwen3:4b").status()
    assert missing.reachable and not missing.model_present

    def refuse(request):
        raise httpx.ConnectError("refused")

    assert not make(refuse).status().reachable
