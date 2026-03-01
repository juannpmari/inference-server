"""Unit tests for MultiTieredCacheManager, REST cache endpoints, and engine wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from data_plane.inference.sidecar.kv_cache_api import KVCacheAPIService
from data_plane.inference.sidecar.l1_cache.api import L1CacheAPI
from data_plane.inference.engine.mock_engine import MockLLMEngine
from data_plane.inference.engine.sidecar_cache_client import SidecarCacheClient
from shared.types import BlockReference, TransferResult


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
    return L1CacheAPI(capacity_bytes=2048, mock=True)


@pytest.fixture
def manager(l1, mock_l2, registry):
    return MultiTieredCacheManager(l1_api=l1, l2_connector=mock_l2, registry=registry)


# --- Cache Manager tests ---


class TestMultiTieredCacheManager:
    @pytest.mark.asyncio
    async def test_offload_stores_in_l1_and_registers(self, manager, registry):
        ref = BlockReference(device_id=0, memory_address=0x1000, size_bytes=256)
        result = await manager.process_offload(ref, "key-1", model_id="m1", prefix_hash="ph1")

        assert result.success is True
        assert "L1" in result.message
        entry = registry.lookup("key-1")
        assert entry is not None
        assert entry.location == "L1"
        assert entry.model_id == "m1"

    @pytest.mark.asyncio
    async def test_fetch_hits_l1_and_updates_access(self, manager, registry):
        ref = BlockReference(device_id=0, memory_address=0x1000, size_bytes=256)
        await manager.process_offload(ref, "key-2")

        dest = BlockReference(device_id=0, memory_address=0x2000, size_bytes=256)
        result = await manager.execute_fetch("key-2", dest)
        assert result.success is True
        assert "L1" in result.message

        entry = registry.lookup("key-2")
        assert entry is not None
        assert entry.access_count >= 1

    @pytest.mark.asyncio
    async def test_fetch_misses_l1_falls_to_l2(self, manager, mock_l2):
        dest = BlockReference(device_id=0, memory_address=0x3000, size_bytes=128)
        result = await manager.execute_fetch("missing-key", dest)
        assert result.success is False
        assert "Miss" in result.message
        mock_l2.get.assert_called_once_with("missing-key")

    @pytest.mark.asyncio
    async def test_check_global_availability(self, manager):
        ref = BlockReference(device_id=0, memory_address=0x1000, size_bytes=128)
        await manager.process_offload(ref, "avail-key")

        assert await manager.check_global_availability("avail-key") is True
        assert await manager.check_global_availability("nope") is False

    @pytest.mark.asyncio
    async def test_execute_l1_eviction(self, manager, registry):
        ref = BlockReference(device_id=0, memory_address=0x1000, size_bytes=512)
        await manager.process_offload(ref, "evict-a")
        await manager.process_offload(
            BlockReference(0, 0x2000, 512), "evict-b"
        )

        result = await manager.execute_l1_eviction(600)
        assert result.success is True
        assert registry.lookup("evict-a") is None  # evicted (LRU)

    @pytest.mark.asyncio
    async def test_l1_full_triggers_eviction_still_succeeds(self, manager):
        # Fill L1 (2048 bytes)
        await manager.process_offload(BlockReference(0, 0x1000, 1024), "fill-a")
        await manager.process_offload(BlockReference(0, 0x2000, 1024), "fill-b")

        # This should trigger eviction of fill-a and succeed
        result = await manager.process_offload(BlockReference(0, 0x3000, 512), "fill-c")
        assert result.success is True


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


# --- Engine wiring tests ---


class TestSidecarCacheClient:
    @pytest.mark.asyncio
    async def test_offload_via_client(self, manager):
        service = KVCacheAPIService(cache_manager=manager)
        client = SidecarCacheClient(kv_service=service)

        ref = BlockReference(device_id=0, memory_address=0x5000, size_bytes=128)
        result = await client.offload_block("wired-key", ref, model_id="m")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_fetch_roundtrip_via_client(self, manager):
        service = KVCacheAPIService(cache_manager=manager)
        client = SidecarCacheClient(kv_service=service)

        # Offload first
        ref = BlockReference(device_id=0, memory_address=0x5000, size_bytes=128)
        await client.offload_block("rt-key", ref)

        # Fetch back
        dest = BlockReference(device_id=0, memory_address=0x6000, size_bytes=128)
        result = await client.fetch_block("rt-key", dest)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_mock_engine_offload(self, manager):
        service = KVCacheAPIService(cache_manager=manager)
        client = SidecarCacheClient(kv_service=service)

        engine = MockLLMEngine()
        engine.set_cache_client(client)

        result = await engine.offload_block("engine-blk", size=256, model_id="test")
        assert result.success is True

        result = await engine.fetch_block("engine-blk", size=256)
        assert result.success is True
