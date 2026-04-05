"""Tests for gateway /ready endpoint with engine health cascading."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from data_plane.gateway import routing


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level state between tests."""
    routing._draining = False
    routing._engine_health_cache["healthy"] = False
    routing._engine_health_cache["checked_at"] = 0.0
    yield
    routing._draining = False
    routing._engine_health_cache["healthy"] = False
    routing._engine_health_cache["checked_at"] = 0.0


@pytest.fixture
def anyio_backend():
    return "asyncio"


class TestReadyEndpoint:

    @pytest.mark.asyncio
    async def test_ready_returns_200_when_engine_healthy(self):
        """When engine /ready returns 200, gateway /ready should return 200."""
        # Simulate healthy engine via the cache
        routing._engine_health_cache["healthy"] = True
        routing._engine_health_cache["checked_at"] = time.monotonic()

        async with AsyncClient(
            transport=_ASGITransport(routing.app), base_url="http://test"
        ) as client:
            resp = await client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_engine_down(self):
        """When engine is unreachable, gateway /ready should return 503."""
        # Force cache to be stale so it re-checks
        routing._engine_health_cache["checked_at"] = 0.0

        # Mock httpx to simulate engine being unreachable
        with patch("data_plane.gateway.routing.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get = AsyncMock(side_effect=Exception("connection refused"))
            MockClient.return_value = mock_instance

            async with AsyncClient(
                transport=_ASGITransport(routing.app), base_url="http://test"
            ) as client:
                resp = await client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["reason"] == "engine_unreachable"

    @pytest.mark.asyncio
    async def test_ready_uses_cache_ttl(self):
        """Cached health result should be reused within TTL."""
        routing._engine_health_cache["healthy"] = True
        routing._engine_health_cache["checked_at"] = time.monotonic()

        # Even if we mock engine as unreachable, cached result should be used
        with patch("data_plane.gateway.routing.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get = AsyncMock(side_effect=Exception("unreachable"))
            MockClient.return_value = mock_instance

            async with AsyncClient(
                transport=_ASGITransport(routing.app), base_url="http://test"
            ) as client:
                resp = await client.get("/ready")
            # Should still be 200 because cache is fresh
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_draining(self):
        """When gateway is draining, /ready should return 503."""
        routing._draining = True
        routing._engine_health_cache["healthy"] = True
        routing._engine_health_cache["checked_at"] = time.monotonic()

        async with AsyncClient(
            transport=_ASGITransport(routing.app), base_url="http://test"
        ) as client:
            resp = await client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["reason"] == "draining"


# ---------------------------------------------------------------------------
# Helper: ASGI transport for testing
# ---------------------------------------------------------------------------

try:
    from httpx import ASGITransport as _ASGITransport
except ImportError:
    # Fallback for older httpx versions
    class _ASGITransport:  # type: ignore[no-redef]
        def __init__(self, app):
            self.app = app
