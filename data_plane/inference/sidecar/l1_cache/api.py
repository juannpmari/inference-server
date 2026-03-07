"""L1 byte store — in-memory block storage for the sidecar.

The sidecar cannot access GPU memory (separate container). It stores raw bytes
keyed by block_id, with LRU eviction and Prometheus metrics.
"""

import logging
import time
from typing import Optional

from data_plane.inference.sidecar.l1_cache import metrics as l1_metrics
from data_plane.inference.sidecar.l1_cache.allocator import BlockSlotAllocator
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy

logger = logging.getLogger(__name__)


class L1ByteStore:
    """In-memory byte store with block-slot allocation and LRU eviction."""

    def __init__(self, num_blocks: int = 1024, block_size_bytes: int = 131072):
        self.allocator = BlockSlotAllocator(num_blocks)
        self.eviction_policy = LRUPolicy()
        self._block_size = block_size_bytes
        # block_id -> bytes
        self._data: dict[int, bytes] = {}
        # block_id -> block_hash (for registry/eviction tracking)
        self._id_to_hash: dict[int, str] = {}
        # block_hash -> block_id (reverse lookup)
        self._hash_to_id: dict[str, int] = {}

        l1_metrics.l1_cache_capacity_bytes.set(num_blocks * block_size_bytes)
        l1_metrics.l1_cache_used_bytes.set(0)
        l1_metrics.l1_cache_blocks_stored.set(0)

    def _update_metrics(self) -> None:
        used = len(self._data) * self._block_size
        cap = self.allocator.num_blocks * self._block_size
        l1_metrics.l1_cache_used_bytes.set(used)
        l1_metrics.l1_cache_utilization_ratio.set(used / cap if cap > 0 else 0)
        l1_metrics.l1_cache_blocks_stored.set(len(self._data))

    def get_num_free_blocks(self) -> int:
        return self.allocator.num_free

    def allocate_blocks(self, block_hashes: list[str]) -> Optional[list[int]]:
        """Reserve block slots for the given hashes. Returns block IDs or None."""
        # Evict if needed to make room
        needed = len(block_hashes) - self.allocator.num_free
        while needed > 0:
            victim_hash = self.eviction_policy.select_victim()
            if not victim_hash:
                break
            victim_id = self._hash_to_id.get(victim_hash)
            if victim_id is not None:
                self._evict(victim_id)
                l1_metrics.l1_cache_evictions_total.labels(reason="capacity").inc()
            needed -= 1

        ids = self.allocator.allocate_n(len(block_hashes))
        if ids is None:
            return None

        for block_id, block_hash in zip(ids, block_hashes):
            self._id_to_hash[block_id] = block_hash
            self._hash_to_id[block_hash] = block_id

        return ids

    def store(self, block_id: int, data: bytes, block_hash: str) -> bool:
        """Store block bytes at a previously allocated slot."""
        start = time.monotonic()
        if not self.allocator.is_allocated(block_id):
            l1_metrics.l1_cache_operations_total.labels(op="store", status="error").inc()
            return False

        self._data[block_id] = data
        self._id_to_hash[block_id] = block_hash
        self._hash_to_id[block_hash] = block_id
        self.eviction_policy.track_new(block_hash, len(data))
        l1_metrics.l1_cache_operations_total.labels(op="store", status="hit").inc()
        l1_metrics.l1_cache_transfer_bytes_total.labels(direction="store").inc(len(data))
        self._update_metrics()
        l1_metrics.l1_cache_operation_duration_seconds.labels(op="store").observe(
            time.monotonic() - start
        )
        return True

    def load(self, block_id: int) -> Optional[bytes]:
        """Retrieve block bytes. Returns None on miss."""
        start = time.monotonic()
        data = self._data.get(block_id)
        if data is None:
            l1_metrics.l1_cache_operations_total.labels(op="load", status="miss").inc()
            return None

        block_hash = self._id_to_hash.get(block_id)
        if block_hash:
            self.eviction_policy.record_access(block_hash)
        l1_metrics.l1_cache_operations_total.labels(op="load", status="hit").inc()
        l1_metrics.l1_cache_transfer_bytes_total.labels(direction="load").inc(len(data))
        l1_metrics.l1_cache_operation_duration_seconds.labels(op="load").observe(
            time.monotonic() - start
        )
        return data

    def free(self, block_id: int) -> bool:
        """Release a block slot and its data."""
        block_hash = self._id_to_hash.pop(block_id, None)
        if block_hash:
            self._hash_to_id.pop(block_hash, None)
            self.eviction_policy.remove(block_hash)
        self._data.pop(block_id, None)
        result = self.allocator.free(block_id)
        self._update_metrics()
        return result

    def _evict(self, block_id: int) -> None:
        """Evict a specific block by ID."""
        block_hash = self._id_to_hash.pop(block_id, None)
        if block_hash:
            self._hash_to_id.pop(block_hash, None)
            self.eviction_policy.remove(block_hash)
        self._data.pop(block_id, None)
        self.allocator.free(block_id)
        self._update_metrics()
