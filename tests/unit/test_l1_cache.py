"""Unit tests for L1 cache components: allocator, LRU policy, L1CacheAPI, KVBlockRegistry."""

import json
import os
import tempfile

import pytest

from data_plane.inference.sidecar.l1_cache.allocator import L1Allocator
from data_plane.inference.sidecar.l1_cache.api import L1CacheAPI
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy
from data_plane.inference.sidecar.l1_cache.gpu_transfer import (
    MockGPUTransferHandler,
    create_transfer_handler,
)
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from shared.types import KVBlockEntry


# --- Allocator tests ---


class TestL1Allocator:
    def test_alloc_free_cycle(self):
        alloc = L1Allocator(capacity_bytes=1024, mock=True)
        ptr = alloc.allocate(256)
        assert ptr is not None
        assert ptr.size_bytes == 256
        assert alloc.used_bytes == 256
        assert alloc.available_bytes == 768

        assert alloc.free(ptr) is True
        assert alloc.used_bytes == 0
        assert alloc.available_bytes == 1024

    def test_capacity_exhaustion(self):
        alloc = L1Allocator(capacity_bytes=512, mock=True)
        ptr1 = alloc.allocate(300)
        assert ptr1 is not None
        ptr2 = alloc.allocate(300)
        assert ptr2 is None  # Not enough space
        assert alloc.used_bytes == 300

    def test_free_list_reclamation(self):
        """After freeing A, a new allocation C should fit in A's space."""
        alloc = L1Allocator(capacity_bytes=1024, mock=True)
        ptr_a = alloc.allocate(256)
        ptr_b = alloc.allocate(256)
        assert ptr_a is not None and ptr_b is not None

        alloc.free(ptr_a)
        assert alloc.available_bytes == 768

        # C should reuse A's freed region
        ptr_c = alloc.allocate(256)
        assert ptr_c is not None
        assert ptr_c.cpu_address == ptr_a.cpu_address  # Same address reused
        assert alloc.used_bytes == 512


# --- LRU Policy tests ---


class TestLRUPolicy:
    def test_ordering_after_mixed_accesses(self):
        lru = LRUPolicy()
        lru.track_new("a", 100)
        lru.track_new("b", 200)
        lru.track_new("c", 300)
        lru.record_access("a")  # a is now most recent

        victim = lru.select_victim()
        assert victim == "b"  # b is oldest

    def test_victim_after_removal(self):
        lru = LRUPolicy()
        lru.track_new("x", 10)
        lru.track_new("y", 20)
        lru.track_new("z", 30)

        lru.remove("x")
        victim = lru.select_victim()
        assert victim == "y"

    def test_empty_returns_none(self):
        lru = LRUPolicy()
        assert lru.select_victim() is None


# --- L1CacheAPI tests ---


class TestL1CacheAPI:
    @pytest.fixture
    def l1(self):
        return L1CacheAPI(capacity_bytes=1024, mock=True)

    @pytest.mark.asyncio
    async def test_put_get_roundtrip(self, l1):
        ok = await l1.put("block-1", hbm_addr=0xAAAA, size=256)
        assert ok is True
        assert l1.has_key("block-1")

        hit = await l1.get("block-1", dest_hbm_addr=0xBBBB)
        assert hit is True

    @pytest.mark.asyncio
    async def test_eviction_on_capacity_pressure(self, l1):
        await l1.put("a", 0x1000, 400)
        await l1.put("b", 0x2000, 400)
        # L1 has 1024 bytes, 800 used. Putting 300 more should evict 'a' (LRU)
        ok = await l1.put("c", 0x3000, 300)
        assert ok is True
        assert not l1.has_key("a")  # evicted
        assert l1.has_key("b")
        assert l1.has_key("c")

    @pytest.mark.asyncio
    async def test_get_miss_returns_false(self, l1):
        hit = await l1.get("nonexistent", dest_hbm_addr=0x9999)
        assert hit is False


# --- GPU Transfer Handler tests ---


class TestGPUTransferHandler:
    def test_factory_returns_mock(self):
        handler = create_transfer_handler(mock=True)
        assert isinstance(handler, MockGPUTransferHandler)

    @pytest.mark.asyncio
    async def test_mock_handler_success(self):
        handler = MockGPUTransferHandler()
        status = await handler.copy_hbm_to_dram(0x1000, 0x2000, 512)
        assert status.success is True

        status = await handler.copy_dram_to_hbm(0x2000, 0x1000, 512)
        assert status.success is True


# --- KV Block Registry tests ---


class TestKVBlockRegistry:
    def test_register_lookup_unregister(self):
        reg = KVBlockRegistry()
        entry = KVBlockEntry(
            key="blk-1", location="L1", size_bytes=4096,
            model_id="test-model", prefix_hash="abc123",
        )
        reg.register(entry)
        assert reg.lookup("blk-1") is not None
        assert reg.lookup("blk-1").model_id == "test-model"

        reg.unregister("blk-1")
        assert reg.lookup("blk-1") is None

    def test_query_by_prefix(self):
        reg = KVBlockRegistry()
        reg.register(KVBlockEntry(key="a", location="L1", size_bytes=100, prefix_hash="pf1", model_id="m1"))
        reg.register(KVBlockEntry(key="b", location="L1", size_bytes=200, prefix_hash="pf1", model_id="m1"))
        reg.register(KVBlockEntry(key="c", location="L1", size_bytes=300, prefix_hash="pf2", model_id="m1"))

        results = reg.query_by_prefix("pf1", "m1")
        assert len(results) == 2
        assert {r.key for r in results} == {"a", "b"}

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kv_blocks.json")
            reg1 = KVBlockRegistry(persist_path=path)
            reg1.register(KVBlockEntry(
                key="persist-test", location="L1", size_bytes=512,
                model_id="m", prefix_hash="ph",
            ))

            # New instance should restore
            reg2 = KVBlockRegistry(persist_path=path)
            restored = reg2.lookup("persist-test")
            assert restored is not None
            assert restored.size_bytes == 512

    def test_stats(self):
        reg = KVBlockRegistry()
        reg.register(KVBlockEntry(key="l1a", location="L1", size_bytes=100))
        reg.register(KVBlockEntry(key="l2a", location="L2", size_bytes=200))

        stats = reg.stats()
        assert stats["total_blocks"] == 2
        assert stats["l1_blocks"] == 1
        assert stats["l2_blocks"] == 1
        assert stats["l1_used_bytes"] == 100
        assert stats["l2_used_bytes"] == 200
