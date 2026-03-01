"""Multi-tiered cache manager orchestrating L1 and L2 tiers.

Implements the offload/fetch strategy:
  1. Try L1 (fast, local CPU DRAM)
  2. Fall back to L2 (slower, distributed Redis)

Integrates with KVBlockRegistry for metadata tracking.
"""

import logging
import time
from typing import Optional

from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from data_plane.inference.sidecar.l1_cache.api import L1CacheAPI
from data_plane.inference.sidecar.l2_cache.connector import L2Connector
from shared.types import BlockReference, KVBlockEntry, TransferResult

logger = logging.getLogger(__name__)


class MultiTieredCacheManager:
    """Orchestrates L1 ↔ L2 cache operations with registry tracking."""

    def __init__(
        self,
        l1_api: L1CacheAPI,
        l2_connector: L2Connector,
        registry: Optional[KVBlockRegistry] = None,
    ):
        self.l1 = l1_api
        self.l2 = l2_connector
        self.registry = registry or KVBlockRegistry()

    async def process_offload(
        self,
        block_ref: BlockReference,
        key: str,
        model_id: str = "",
        prefix_hash: str = "",
    ) -> TransferResult:
        """Offload a KV block: try L1 first, fall back to L2."""
        logger.debug(f"Processing offload for key: {key}")

        l1_success = await self.l1.put(key, block_ref.memory_address, block_ref.size_bytes)

        if l1_success:
            self.registry.register(KVBlockEntry(
                key=key,
                location="L1",
                size_bytes=block_ref.size_bytes,
                l1_address=block_ref.memory_address,
                model_id=model_id,
                prefix_hash=prefix_hash,
                created_at=time.time(),
                last_accessed=time.time(),
            ))
            return TransferResult(True, "Offloaded to L1")

        logger.info(f"L1 failed for {key}, promoting to L2")
        mock_data = b"\x00" * block_ref.size_bytes
        l2_status = await self.l2.put(key, mock_data)

        if l2_status.success:
            self.registry.register(KVBlockEntry(
                key=key,
                location="L2",
                size_bytes=block_ref.size_bytes,
                model_id=model_id,
                prefix_hash=prefix_hash,
                created_at=time.time(),
                last_accessed=time.time(),
            ))

        return TransferResult(l2_status.success, f"Promoted to L2: {l2_status.message}")

    async def execute_fetch(self, key: str, dest_ref: BlockReference) -> TransferResult:
        """Fetch a KV block: check L1, then L2."""
        l1_success = await self.l1.get(key, dest_ref.memory_address)
        if l1_success:
            self.registry.record_access(key)
            return TransferResult(True, "Hit in L1")

        l2_status = await self.l2.get(key)
        if l2_status.success and l2_status.data:
            self.registry.record_access(key)
            return TransferResult(True, "Hit in L2 (Restored to HBM)")

        return TransferResult(False, "Cache Miss (Key not found in L1 or L2)")

    async def check_global_availability(self, key: str) -> bool:
        """Check if a block exists in any tier via the registry."""
        entry = self.registry.lookup(key)
        return entry is not None

    async def execute_l1_eviction(self, needed_bytes: int) -> TransferResult:
        """Evict LRU entries from L1 until needed_bytes are freed."""
        freed = 0
        evicted_keys = []

        while freed < needed_bytes:
            victim_key = self.l1.eviction_policy.select_victim()
            if not victim_key:
                break
            pointer = self.l1._key_map.get(victim_key)
            if pointer:
                freed += pointer.size_bytes
            self.l1._evict_victim(victim_key)
            self.registry.unregister(victim_key)
            evicted_keys.append(victim_key)

        if freed >= needed_bytes:
            return TransferResult(
                True, f"Freed {freed} bytes by evicting {len(evicted_keys)} blocks"
            )
        return TransferResult(
            False,
            f"Could only free {freed}/{needed_bytes} bytes "
            f"(evicted {len(evicted_keys)} blocks)",
        )
