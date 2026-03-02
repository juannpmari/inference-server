"""Unit tests for byte-oriented MultiTieredCacheManager and REST cache endpoints."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from data_plane.inference.sidecar.l1_cache.api import L1ByteStore
from shared.types import TransferResult


@pytest.fixture
def mock_l2():
    l2 = AsyncMock()
    l2.put = AsyncMock(return_value=TransferResult(True, "L2 OK"))
    l2.get = AsyncMock(return_value=TransferResult(False, "L2 miss"))
    return l2


@pytest.fixture
def registry():
    return KVBlockRegistry()


@pytest.fixture
def l1():
    return L1ByteStore(num_blocks=8, block_size_bytes=64)


@pytest.fixture
def manager(l1, mock_l2, registry):
    return MultiTieredCacheManager(l1=l1, l2=mock_l2, registry=registry)


class TestMultiTieredCacheManager:
    @pytest.mark.asyncio
    async def test_store_and_load_bytes(self, manager, registry):
        ids = manager.allocate_blocks(["hash-1"])
        assert ids is not None
        block_id = ids[0]

        data = b"test-data" + b"\x00" * 55  # 64 bytes
        ok = await manager.store_block(block_id, "hash-1", data, model_id="m1")
        assert ok is True

        entry = registry.lookup("hash-1")
        assert entry is not None
        assert entry.location == "L1"
        assert entry.model_id == "m1"

        loaded = await manager.load_block(block_id)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_load_miss(self, manager):
        result = await manager.load_block(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_free_block(self, manager, registry):
        ids = manager.allocate_blocks(["free-hash"])
        assert ids is not None
        block_id = ids[0]

        data = b"\x01" * 64
        await manager.store_block(block_id, "free-hash", data)

        ok = manager.free_block(block_id)
        assert ok is True
        assert registry.lookup("free-hash") is None

        # Slot should be returned to pool
        assert manager.get_num_free_blocks() == 8

    @pytest.mark.asyncio
    async def test_get_num_free_blocks(self, manager):
        assert manager.get_num_free_blocks() == 8

        manager.allocate_blocks(["h1", "h2"])
        assert manager.get_num_free_blocks() == 6

    @pytest.mark.asyncio
    async def test_allocate_blocks_returns_ids(self, manager):
        ids = manager.allocate_blocks(["a", "b", "c"])
        assert ids is not None
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique


# --- REST endpoint tests ---


class TestCacheRESTEndpoints:
    @pytest.mark.asyncio
    async def test_cache_blocks_empty(self):
        from data_plane.inference.sidecar.api import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/cache/blocks")
            assert r.status_code == 200
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_cache_stats_structure(self):
        from data_plane.inference.sidecar.api import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/cache/stats")
            assert r.status_code == 200
            body = r.json()
            assert "total_blocks" in body
            assert "hit_rate" in body
            assert "l1_blocks" in body
