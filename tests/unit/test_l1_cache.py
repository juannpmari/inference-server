"""Unit tests for L1 byte store, block slot allocator, LRU policy, and KVBlockRegistry."""

import json
import os
import tempfile

import pytest

from data_plane.inference.sidecar.l1_cache.allocator import BlockSlotAllocator
from data_plane.inference.sidecar.l1_cache.api import L1ByteStore
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from shared.types import KVBlockEntry


# --- BlockSlotAllocator tests ---


class TestBlockSlotAllocator:
    def test_allocate_free_cycle(self):
        alloc = BlockSlotAllocator(num_blocks=4)
        bid = alloc.allocate()
        assert bid is not None
        assert alloc.num_allocated == 1
        assert alloc.num_free == 3

        assert alloc.free(bid) is True
        assert alloc.num_allocated == 0
        assert alloc.num_free == 4

    def test_capacity_exhaustion(self):
        alloc = BlockSlotAllocator(num_blocks=2)
        id1 = alloc.allocate()
        id2 = alloc.allocate()
        assert id1 is not None and id2 is not None
        assert alloc.num_free == 0

        id3 = alloc.allocate()
        assert id3 is None

    def test_allocate_n(self):
        alloc = BlockSlotAllocator(num_blocks=4)
        ids = alloc.allocate_n(3)
        assert ids is not None
        assert len(ids) == 3
        assert alloc.num_free == 1

        # Not enough for 2 more
        assert alloc.allocate_n(2) is None

    def test_free_reuses_id(self):
        alloc = BlockSlotAllocator(num_blocks=2)
        id1 = alloc.allocate()
        id2 = alloc.allocate()
        alloc.free(id1)

        id3 = alloc.allocate()
        assert id3 == id1  # reuses freed ID


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


# --- L1ByteStore tests ---


class TestL1ByteStore:
    @pytest.fixture
    def store(self):
        return L1ByteStore(num_blocks=4, block_size_bytes=64)

    def test_store_load_roundtrip(self, store):
        ids = store.allocate_blocks(["hash-1"])
        assert ids is not None
        block_id = ids[0]

        data = b"hello world" + b"\x00" * 53  # 64 bytes
        assert store.store(block_id, data, "hash-1") is True

        loaded = store.load(block_id)
        assert loaded == data

    def test_capacity_exhaustion(self, store):
        hashes = [f"h{i}" for i in range(4)]
        ids = store.allocate_blocks(hashes)
        assert ids is not None
        # Store data so LRU tracks them for eviction
        for bid, h in zip(ids, hashes):
            store.store(bid, b"\x00" * 64, h)
        assert store.get_num_free_blocks() == 0

        # Eviction should free a slot
        assert store.allocate_blocks(["extra"]) is not None

    def test_eviction_frees_lru(self, store):
        # Fill all 4 blocks
        hashes = [f"h{i}" for i in range(4)]
        ids = store.allocate_blocks(hashes)
        for bid, h in zip(ids, hashes):
            store.store(bid, b"\x00" * 64, h)

        assert store.get_num_free_blocks() == 0

        # Allocate one more — should evict LRU
        new_ids = store.allocate_blocks(["h-new"])
        assert new_ids is not None
        assert store.get_num_free_blocks() == 0  # used the freed slot

    def test_allocate_free_cycle(self, store):
        ids = store.allocate_blocks(["hash-a"])
        assert ids is not None
        bid = ids[0]

        store.store(bid, b"\x00" * 64, "hash-a")
        assert store.free(bid) is True
        assert store.get_num_free_blocks() == 4

        # Reuse the freed slot
        ids2 = store.allocate_blocks(["hash-b"])
        assert ids2 is not None

    def test_load_miss(self, store):
        assert store.load(999) is None


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
