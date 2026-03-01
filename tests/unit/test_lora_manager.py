"""
Unit tests for LoRAManager: adapter lifecycle, deduplication, LRU eviction.

Uses a minimal mock engine (add_lora/remove_lora tracking) and
mocked httpx responses to simulate the sidecar.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest

from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine.lora_manager import LoRAManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Minimal mock with add_lora/remove_lora tracking (sync, like real vLLM)."""
    engine = MagicMock()
    engine._loaded = {}

    def _add_lora(lora_req):
        engine._loaded[lora_req.lora_int_id] = lora_req

    def _remove_lora(lora_int_id):
        engine._loaded.pop(lora_int_id, None)

    engine.add_lora = MagicMock(side_effect=_add_lora)
    engine.remove_lora = MagicMock(side_effect=_remove_lora)
    return engine


@pytest.fixture
def lora_config():
    return EngineConfig(
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        adapter_poll_interval=0.01,  # fast polling for tests
        adapter_poll_timeout=5.0,
    )


@pytest.fixture
def manager(mock_engine, lora_config):
    return LoRAManager(
        engine=mock_engine,
        config=lora_config,
        sidecar_url="http://mock-sidecar:8001",
    )


def _make_httpx_response(status_code: int, json_data: dict) -> httpx.Response:
    """Build a real httpx.Response for mocking."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://mock"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFirstLoad:
    """First load triggers sidecar + add_lora, returns valid LoRARequest."""

    @pytest.mark.asyncio
    async def test_first_load_calls_sidecar_and_loads(self, manager, mock_engine):
        # Sidecar returns 200 (already cached on disk)
        trigger_resp = _make_httpx_response(200, {
            "status": "loaded",
            "adapter_identifier": "org/adapter-a",
            "local_path": "/mnt/models/org--adapter-a/latest",
        })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=trigger_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.ensure_adapter_loaded("org/adapter-a", "latest")

        assert result.lora_name == "org/adapter-a-latest"
        assert result.lora_path == "/mnt/models/org--adapter-a/latest"
        mock_engine.add_lora.assert_called_once()
        assert manager.loaded_count == 1


class TestCacheHit:
    """Second call for same adapter is a cache hit — no sidecar, no add_lora."""

    @pytest.mark.asyncio
    async def test_second_load_is_cache_hit(self, manager, mock_engine):
        trigger_resp = _make_httpx_response(200, {
            "status": "loaded",
            "adapter_identifier": "org/adapter-a",
            "local_path": "/mnt/models/org--adapter-a/latest",
        })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=trigger_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await manager.ensure_adapter_loaded("org/adapter-a", "latest")
            # Reset call counts
            mock_engine.add_lora.reset_mock()
            mock_client.post.reset_mock()

            result = await manager.ensure_adapter_loaded("org/adapter-a", "latest")

        assert result.lora_name == "org/adapter-a-latest"
        mock_engine.add_lora.assert_not_called()
        mock_client.post.assert_not_called()


class TestConcurrentDedup:
    """Multiple concurrent requests for same adapter → one sidecar call."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_deduplicate(self, manager, mock_engine):
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate a short download delay
            await asyncio.sleep(0.05)
            return _make_httpx_response(200, {
                "status": "loaded",
                "adapter_identifier": "org/adapter-a",
                "local_path": "/mnt/models/org--adapter-a/latest",
            })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await asyncio.gather(*[
                manager.ensure_adapter_loaded("org/adapter-a", "latest")
                for _ in range(5)
            ])

        assert len(results) == 5
        assert all(r.lora_name == "org/adapter-a-latest" for r in results)
        assert call_count == 1  # Only one sidecar call
        assert mock_engine.add_lora.call_count == 1


class TestLRUEviction:
    """With max_loras=2, loading 3rd adapter evicts the oldest."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_full(self, manager, mock_engine):
        async def mock_post(url, **kwargs):
            # Extract adapter name from URL
            adapter_id = url.split("/adapter/load/")[1]
            return _make_httpx_response(200, {
                "status": "loaded",
                "adapter_identifier": adapter_id,
                "local_path": f"/mnt/models/{adapter_id}/latest",
            })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await manager.ensure_adapter_loaded("adapter-a", "latest")
            await manager.ensure_adapter_loaded("adapter-b", "latest")
            assert manager.loaded_count == 2
            mock_engine.remove_lora.assert_not_called()

            await manager.ensure_adapter_loaded("adapter-c", "latest")

        assert manager.loaded_count == 2
        mock_engine.remove_lora.assert_called_once()
        # adapter-a was evicted (oldest)
        assert "adapter-a@latest" not in manager.loaded_keys
        assert "adapter-b@latest" in manager.loaded_keys
        assert "adapter-c@latest" in manager.loaded_keys


class TestLRUOrdering:
    """Accessing adapter A moves it to back; B gets evicted instead."""

    @pytest.mark.asyncio
    async def test_access_refreshes_lru(self, manager, mock_engine):
        async def mock_post(url, **kwargs):
            adapter_id = url.split("/adapter/load/")[1]
            return _make_httpx_response(200, {
                "status": "loaded",
                "adapter_identifier": adapter_id,
                "local_path": f"/mnt/models/{adapter_id}/latest",
            })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await manager.ensure_adapter_loaded("adapter-a", "latest")
            await manager.ensure_adapter_loaded("adapter-b", "latest")
            # Touch A so B becomes the LRU
            await manager.ensure_adapter_loaded("adapter-a", "latest")

            await manager.ensure_adapter_loaded("adapter-c", "latest")

        # B was evicted (LRU), not A
        assert "adapter-a@latest" in manager.loaded_keys
        assert "adapter-b@latest" not in manager.loaded_keys
        assert "adapter-c@latest" in manager.loaded_keys


class TestErrorPropagation:
    """Sidecar failure raises RuntimeError."""

    @pytest.mark.asyncio
    async def test_sidecar_error_propagates(self, manager, mock_engine):
        error_resp = _make_httpx_response(500, {"detail": "Internal error"})

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=error_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await manager.ensure_adapter_loaded("bad/adapter", "latest")

        mock_engine.add_lora.assert_not_called()
        assert manager.loaded_count == 0


class TestErrorIsolation:
    """Failed load for X doesn't block subsequent load for Y."""

    @pytest.mark.asyncio
    async def test_failure_does_not_block_others(self, manager, mock_engine):
        call_num = 0

        async def mock_post(url, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return _make_httpx_response(500, {"detail": "fail"})
            adapter_id = url.split("/adapter/load/")[1]
            return _make_httpx_response(200, {
                "status": "loaded",
                "adapter_identifier": adapter_id,
                "local_path": f"/mnt/models/{adapter_id}/latest",
            })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await manager.ensure_adapter_loaded("bad/adapter", "latest")

            result = await manager.ensure_adapter_loaded("good/adapter", "latest")

        assert result.lora_name == "good/adapter-latest"
        assert manager.loaded_count == 1


class TestPollingFlow:
    """When sidecar returns 202, manager polls registry until loaded."""

    @pytest.mark.asyncio
    async def test_polls_until_loaded(self, manager, mock_engine):
        trigger_resp = _make_httpx_response(202, {
            "status": "downloading",
            "adapter_identifier": "org/adapter-a",
        })

        poll_count = 0

        async def mock_get(url, **kwargs):
            nonlocal poll_count
            poll_count += 1
            if poll_count < 3:
                return _make_httpx_response(200, {
                    "org/adapter-a": {
                        "adapter_id": "org/adapter-a",
                        "status": "downloading",
                    }
                })
            return _make_httpx_response(200, {
                "org/adapter-a": {
                    "adapter_id": "org/adapter-a",
                    "status": "loaded",
                    "local_path": "/mnt/models/org--adapter-a/latest",
                }
            })

        with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=trigger_resp)
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.ensure_adapter_loaded("org/adapter-a", "latest")

        assert result.lora_path == "/mnt/models/org--adapter-a/latest"
        assert poll_count == 3
        mock_engine.add_lora.assert_called_once()
