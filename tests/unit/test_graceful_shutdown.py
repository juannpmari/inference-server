"""Tests for graceful shutdown with request draining."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from data_plane.inference.engine.mock_engine import MockLLMEngine
from data_plane.inference.engine.config import EngineConfig


# ---------------------------------------------------------------------------
# MockLLMEngine.in_flight_count
# ---------------------------------------------------------------------------

def test_in_flight_count_empty():
    engine = MockLLMEngine(config=EngineConfig())
    assert engine.in_flight_count == 0


def test_in_flight_count_with_futures():
    engine = MockLLMEngine(config=EngineConfig())
    engine.request_futures["0"] = asyncio.Future()
    engine.request_futures["1"] = asyncio.Future()
    assert engine.in_flight_count == 2


def test_in_flight_count_with_queues():
    engine = MockLLMEngine(config=EngineConfig())
    engine.request_queues["0"] = asyncio.Queue()
    assert engine.in_flight_count == 1


def test_in_flight_count_mixed():
    engine = MockLLMEngine(config=EngineConfig())
    engine.request_futures["0"] = asyncio.Future()
    engine.request_queues["1"] = asyncio.Queue()
    assert engine.in_flight_count == 2


# ---------------------------------------------------------------------------
# Engine API drain behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
def _engine_api():
    """Import engine api module and reset its global state for each test."""
    import data_plane.inference.engine.api as api_mod

    # Save original state
    orig_engine = api_mod._engine
    orig_draining = api_mod._draining
    orig_config = api_mod._config

    # Set up a mock engine and config
    cfg = EngineConfig()
    mock_engine = MockLLMEngine(config=cfg)
    api_mod._engine = mock_engine
    api_mod._config = cfg
    api_mod._draining = False

    yield api_mod

    # Restore original state
    api_mod._engine = orig_engine
    api_mod._draining = orig_draining
    api_mod._config = orig_config


def test_503_when_draining(_engine_api):
    """New requests get 503 during drain."""
    api_mod = _engine_api
    api_mod._draining = True

    client = TestClient(api_mod.app, raise_server_exceptions=False)
    resp = client.post("/generate", json={"prompt": "hello", "max_tokens": 10})
    assert resp.status_code == 503
    assert "Shutting down" in resp.json()["detail"]
    assert resp.headers.get("retry-after") == "1"


def test_health_200_when_draining(_engine_api):
    """/healthz must stay 200 during drain (K8s liveness)."""
    api_mod = _engine_api
    api_mod._draining = True

    client = TestClient(api_mod.app, raise_server_exceptions=False)
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_ready_503_when_draining(_engine_api):
    """/readyz returns 503 during drain."""
    api_mod = _engine_api
    api_mod._draining = True

    client = TestClient(api_mod.app, raise_server_exceptions=False)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["reason"] == "draining"


def test_ready_200_when_not_draining(_engine_api):
    """/readyz returns 200 when not draining and engine is ready."""
    api_mod = _engine_api
    api_mod._draining = False

    client = TestClient(api_mod.app, raise_server_exceptions=False)
    resp = client.get("/readyz")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Drain waits for in-flight requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_waits_for_in_flight():
    """Drain loop waits until in_flight_count reaches 0."""
    cfg = EngineConfig(drain_timeout=5.0)
    engine = MockLLMEngine(config=cfg)

    # Simulate an in-flight future
    fut = asyncio.Future()
    engine.request_futures["0"] = fut
    assert engine.in_flight_count == 1

    drain_done = asyncio.Event()

    async def drain_loop():
        deadline = asyncio.get_event_loop().time() + cfg.drain_timeout
        while engine.in_flight_count > 0:
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(0.05)
        drain_done.set()

    task = asyncio.create_task(drain_loop())

    # After a short delay, resolve the in-flight request
    await asyncio.sleep(0.1)
    fut.set_result("done")
    del engine.request_futures["0"]

    await asyncio.wait_for(drain_done.wait(), timeout=2.0)
    assert engine.in_flight_count == 0
    await task


@pytest.mark.asyncio
async def test_drain_timeout_cancels_remaining():
    """When drain timeout is reached, drain loop exits with remaining requests."""
    cfg = EngineConfig(drain_timeout=0.2)
    engine = MockLLMEngine(config=cfg)

    # Simulate a stuck in-flight future
    fut = asyncio.Future()
    engine.request_futures["0"] = fut

    timed_out = False
    remaining_at_timeout = 0

    async def drain_loop():
        nonlocal timed_out, remaining_at_timeout
        deadline = asyncio.get_event_loop().time() + cfg.drain_timeout
        while engine.in_flight_count > 0:
            if asyncio.get_event_loop().time() >= deadline:
                timed_out = True
                remaining_at_timeout = engine.in_flight_count
                break
            await asyncio.sleep(0.05)

    await drain_loop()
    assert timed_out is True
    assert remaining_at_timeout == 1

    # Cleanup
    fut.cancel()


# ---------------------------------------------------------------------------
# Gateway drain behaviour
# ---------------------------------------------------------------------------

def test_gateway_503_when_draining():
    """Gateway returns 503 on new requests when draining."""
    import data_plane.gateway.routing as gw_mod

    orig_draining = gw_mod._draining
    gw_mod._draining = True

    try:
        client = TestClient(gw_mod.app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/completions",
            json={"model": "Qwen/Qwen2-0.5B-Instruct", "prompt": "hello"},
        )
        assert resp.status_code == 503
    finally:
        gw_mod._draining = orig_draining


def test_gateway_ready_503_when_draining():
    """Gateway /readyz returns 503 when draining."""
    import data_plane.gateway.routing as gw_mod

    orig_draining = gw_mod._draining
    gw_mod._draining = True

    try:
        client = TestClient(gw_mod.app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert resp.json()["reason"] == "draining"
    finally:
        gw_mod._draining = orig_draining


def test_gateway_health_200_when_draining():
    """Gateway /healthz stays 200 even when draining."""
    import data_plane.gateway.routing as gw_mod

    orig_draining = gw_mod._draining
    gw_mod._draining = True

    try:
        client = TestClient(gw_mod.app, raise_server_exceptions=False)
        resp = client.get("/healthz")
        assert resp.status_code == 200
    finally:
        gw_mod._draining = orig_draining


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_engine_drain_timeout_default():
    cfg = EngineConfig()
    assert cfg.drain_timeout == 30.0


def test_engine_drain_timeout_env(monkeypatch):
    monkeypatch.setenv("ENGINE_DRAIN_TIMEOUT", "60")
    cfg = EngineConfig()
    assert cfg.drain_timeout == 60.0


def test_gateway_drain_timeout_default():
    from data_plane.gateway.config import GatewayConfig
    cfg = GatewayConfig()
    assert cfg.drain_timeout == 30.0


def test_gateway_drain_timeout_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_DRAIN_TIMEOUT", "45")
    from data_plane.gateway.config import GatewayConfig
    cfg = GatewayConfig()
    assert cfg.drain_timeout == 45.0
