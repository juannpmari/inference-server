"""
Tests for OpenAI-compatible API endpoints.
Covers Pydantic models, gateway endpoints with mocked engine, and SSE streaming.
"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
from fastapi.testclient import TestClient

from shared.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChunk,
    ChatCompletionChunk,
    ModelObject,
    ModelListResponse,
    OpenAIErrorResponse,
    Usage,
    generate_completion_id,
    now_unix,
)


# ---------------------------------------------------------------------------
# Pydantic model serialization tests
# ---------------------------------------------------------------------------


class TestOpenAITypes:
    def test_chat_message_serialization(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.model_dump()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.stream is False
        assert req.n == 1

    def test_completion_request_defaults(self):
        req = CompletionRequest(model="test", prompt="Hello")
        assert req.temperature == 1.0
        assert req.stream is False

    def test_completion_response_shape(self):
        resp = CompletionResponse(
            id="cmpl-abc",
            created=1234,
            model="test",
            choices=[{"index": 0, "text": "world", "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        d = resp.model_dump()
        assert d["object"] == "text_completion"
        assert d["choices"][0]["text"] == "world"

    def test_chat_completion_response_shape(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-abc",
            created=1234,
            model="test",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }],
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
        d = resp.model_dump()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["role"] == "assistant"

    def test_model_list_response(self):
        resp = ModelListResponse(data=[
            ModelObject(id="m1", created=0, owned_by="org"),
        ])
        d = resp.model_dump()
        assert d["object"] == "list"
        assert len(d["data"]) == 1
        assert d["data"][0]["id"] == "m1"

    def test_error_response(self):
        resp = OpenAIErrorResponse(error={
            "message": "not found",
            "type": "not_found_error",
            "param": None,
            "code": None,
        })
        d = resp.model_dump()
        assert d["error"]["message"] == "not found"

    def test_generate_completion_id(self):
        cid = generate_completion_id("chatcmpl")
        assert cid.startswith("chatcmpl-")
        assert len(cid) > 10

    def test_streaming_chunk_models(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-x",
            created=0,
            model="test",
            choices=[{"index": 0, "delta": {"content": "tok"}}],
        )
        d = chunk.model_dump()
        assert d["object"] == "chat.completion.chunk"

        comp_chunk = CompletionChunk(
            id="cmpl-x",
            created=0,
            model="test",
            choices=[{"index": 0, "text": "tok"}],
        )
        d2 = comp_chunk.model_dump()
        assert d2["object"] == "text_completion"


# ---------------------------------------------------------------------------
# Gateway endpoint tests (using mocked httpx transport)
# ---------------------------------------------------------------------------


def _make_engine_response(text="Hello!", prompt_tokens=5, tokens_generated=3, finish_reason="stop"):
    return {
        "text": text,
        "tokens_generated": tokens_generated,
        "duration_seconds": 0.1,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
    }


def _make_template_response(prompt="user: Hello\nassistant:"):
    return {"prompt": prompt}


class MockTransport(httpx.AsyncBaseTransport):
    """Mock httpx transport that returns canned responses for engine endpoints."""

    def __init__(self):
        self.calls = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        url = str(request.url)
        body = json.loads(request.content) if request.content else {}

        if "/chat/apply_template" in url:
            return httpx.Response(200, json=_make_template_response())
        if "/generate/stream" in url:
            # Return SSE-formatted body
            lines = (
                'data: {"token": "Hello", "finish_reason": null}\n\n'
                'data: {"token": " world", "finish_reason": null}\n\n'
                'data: {"token": "", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 2}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(200, text=lines, headers={"content-type": "text/event-stream"})
        if "/generate" in url:
            return httpx.Response(200, json=_make_engine_response())

        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def gateway_client():
    """Create a TestClient for the gateway app with mocked httpx clients."""
    from data_plane.gateway.routing import app

    transport = MockTransport()
    mock_client = httpx.AsyncClient(transport=transport)

    with TestClient(app) as client:
        # Patch the http clients used by the gateway
        app.state.http_client = mock_client
        app.state.stream_client = mock_client
        yield client


class TestGatewayModels:
    def test_list_models(self, gateway_client):
        resp = gateway_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["object"] == "model"

    def test_get_model(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]
        resp = gateway_client.get(f"/v1/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == model_id

    def test_get_model_not_found(self, gateway_client):
        resp = gateway_client.get("/v1/models/nonexistent-model")
        assert resp.status_code == 404
        assert "error" in resp.json()


class TestGatewayCompletions:
    def test_completions_non_streaming(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        resp = gateway_client.post("/v1/completions", json={
            "model": model_id,
            "prompt": "Hello",
            "max_tokens": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "Hello!"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 5

    def test_completions_model_not_found(self, gateway_client):
        resp = gateway_client.post("/v1/completions", json={
            "model": "nonexistent",
            "prompt": "Hello",
        })
        assert resp.status_code == 404
        assert "error" in resp.json()


class TestGatewayChatCompletions:
    def test_chat_completions_non_streaming(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        resp = gateway_client.post("/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["id"].startswith("chatcmpl-")

    def test_chat_completions_with_sampling_params(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        resp = gateway_client.post("/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "seed": 42,
        })
        assert resp.status_code == 200

    def test_chat_completions_tool_calls_rejected(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        resp = gateway_client.post("/v1/chat/completions", json={
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": "Hello",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            }],
        })
        assert resp.status_code == 400


class TestGatewayErrorFormat:
    def test_404_is_openai_format(self, gateway_client):
        resp = gateway_client.get("/v1/models/does-not-exist")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]

    def test_health_still_works(self, gateway_client):
        resp = gateway_client.get("/health")
        assert resp.status_code == 200


class TestGatewayStreaming:
    def test_completions_streaming(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        with gateway_client.stream("POST", "/v1/completions", json={
            "model": model_id,
            "prompt": "Hello",
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        break
                    chunks.append(json.loads(raw))
            assert len(chunks) >= 1
            assert chunks[0]["object"] == "text_completion"

    def test_chat_completions_streaming(self, gateway_client):
        from data_plane.gateway.routing import MODEL_SERVICE_MAP
        model_id = list(MODEL_SERVICE_MAP.keys())[0]

        with gateway_client.stream("POST", "/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        break
                    chunks.append(json.loads(raw))
            # First chunk should have role
            assert chunks[0]["object"] == "chat.completion.chunk"
            assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"


# ---------------------------------------------------------------------------
# Engine API endpoint tests
# ---------------------------------------------------------------------------


class TestEngineEndpoints:
    """Test the engine-level /generate and /chat/apply_template endpoints."""

    @pytest.fixture
    def engine_client(self, tmp_path):
        """TestClient for engine app with mock engine injected, bypassing lifespan."""
        from contextlib import asynccontextmanager
        from data_plane.inference.engine import api as engine_api
        from data_plane.inference.engine.mock_engine import MockLLMEngine
        from data_plane.inference.engine.config import EngineConfig

        config = EngineConfig(
            enable_engine_mock=True,
            monitoring_local_store_path=str(tmp_path / "metrics.jsonl"),
        )
        mock = MockLLMEngine(config)

        # Inject into module globals
        engine_api._engine = mock
        engine_api._config = config

        # Replace lifespan to avoid /mnt/models permission issue
        original_lifespan = engine_api.app.router.lifespan_context

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        engine_api.app.router.lifespan_context = _noop_lifespan

        with TestClient(engine_api.app) as client:
            yield client

        # Cleanup
        engine_api._engine = None
        engine_api.app.router.lifespan_context = original_lifespan

    def test_generate_endpoint(self, engine_client):
        resp = engine_client.post("/generate", json={"prompt": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "prompt_tokens" in data
        assert "finish_reason" in data

    def test_generate_with_new_params(self, engine_client):
        resp = engine_client.post("/generate", json={
            "prompt": "test",
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "seed": 42,
        })
        assert resp.status_code == 200

    def test_chat_apply_template(self, engine_client):
        resp = engine_client.post("/chat/apply_template", json={
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data
        assert "user: Hi" in data["prompt"]

    def test_generate_stream_endpoint(self, engine_client):
        with engine_client.stream("POST", "/generate/stream", json={"prompt": "hello"}) as resp:
            assert resp.status_code == 200
            lines = list(resp.iter_lines())
            # Should have data lines and end with [DONE]
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert len(data_lines) >= 2
            assert data_lines[-1] == "data: [DONE]"
