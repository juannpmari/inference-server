"""E2E test: full inference request flow through gateway → engine → sidecar.

Requires the test docker-compose stack to be running with mock engine enabled:
    docker compose -f docker-compose.yml \\
        -f tests/integration/docker-compose.test.yml up -d
"""

import httpx
import pytest

GATEWAY_URL = "http://localhost:8000"


@pytest.fixture(scope="module", autouse=True)
def _ensure_stack_ready():
    """Skip all tests if the stack is not running and ready."""
    try:
        resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=10.0)
        if resp.status_code != 200:
            pytest.skip("Stack not ready (gateway /readyz != 200)")
    except httpx.HTTPError:
        pytest.skip("Stack not running (gateway unreachable)")


class TestChatCompletions:
    """POST /v1/chat/completions — non-streaming."""

    def test_basic_chat_completion(self):
        payload = {
            "model": "Qwen/Qwen2-0.5B",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 32,
            "stream": False,
        }
        resp = httpx.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            timeout=60.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        # OpenAI-compatible response schema
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) >= 1
        choice = data["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert "usage" in data

    def test_chat_completion_returns_request_id(self):
        payload = {
            "model": "Qwen/Qwen2-0.5B",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
        }
        resp = httpx.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            headers={"X-Request-ID": "test-req-001"},
            timeout=60.0,
        )
        assert resp.headers.get("x-request-id") == "test-req-001"


class TestCompletions:
    """POST /v1/completions — non-streaming."""

    def test_basic_completion(self):
        payload = {
            "model": "Qwen/Qwen2-0.5B",
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "stream": False,
        }
        resp = httpx.post(
            f"{GATEWAY_URL}/v1/completions",
            json=payload,
            timeout=60.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) >= 1
        assert isinstance(data["choices"][0]["text"], str)


class TestStreamingChatCompletion:
    """POST /v1/chat/completions with stream=True."""

    def test_streaming_chat_returns_sse(self):
        payload = {
            "model": "Qwen/Qwen2-0.5B",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "max_tokens": 32,
            "stream": True,
        }
        chunks = []
        with httpx.stream(
            "POST",
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            timeout=60.0,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line)
        assert len(chunks) > 0, "Should have received at least one SSE chunk"


class TestModelList:
    """GET /v1/models."""

    def test_list_models(self):
        resp = httpx.get(f"{GATEWAY_URL}/v1/models", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
