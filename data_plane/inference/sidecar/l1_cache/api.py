"""L1 Cache facade coordinating allocator, transfer handler, and eviction policy."""

import logging
import time
from typing import Dict, Optional

from data_plane.inference.sidecar.l1_cache import metrics as l1_metrics
from data_plane.inference.sidecar.l1_cache.allocator import L1Allocator
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy
from data_plane.inference.sidecar.l1_cache.gpu_transfer import (
    GPUTransferHandler,
    create_transfer_handler,
)
from shared.types import AllocationPointer

logger = logging.getLogger(__name__)


class L1CacheAPI:
    """Facade providing simple put/get over allocator + transfer + eviction."""

    def __init__(
        self,
        capacity_bytes: int = 512 * 1024 * 1024,
        transfer_handler: Optional[GPUTransferHandler] = None,
        mock: bool = True,
    ):
        self.allocator = L1Allocator(capacity_bytes, mock=mock)
        self.transfer = transfer_handler or create_transfer_handler(mock=mock)
        self.eviction_policy = LRUPolicy()
        self._key_map: Dict[str, AllocationPointer] = {}
        l1_metrics.l1_cache_capacity_bytes.set(capacity_bytes)
        l1_metrics.l1_cache_used_bytes.set(0)
        l1_metrics.l1_cache_blocks_stored.set(0)

    def _update_capacity_metrics(self) -> None:
        used = self.allocator.used_bytes
        cap = self.allocator.capacity_bytes
        l1_metrics.l1_cache_used_bytes.set(used)
        l1_metrics.l1_cache_utilization_ratio.set(used / cap if cap > 0 else 0)
        l1_metrics.l1_cache_blocks_stored.set(len(self._key_map))

    async def put(self, key: str, hbm_addr: int, size: int) -> bool:
        """Offload data from HBM to L1. Handles allocation and automatic eviction."""
        start = time.monotonic()
        pointer = self.allocator.allocate(size)

        while pointer is None:
            victim_key = self.eviction_policy.select_victim()
            if not victim_key:
                logger.error("L1 full and no victims to evict")
                l1_metrics.l1_cache_operations_total.labels(op="put", status="error").inc()
                return False

            logger.debug(f"Evicting L1 victim: {victim_key}")
            self._evict_victim(victim_key)
            l1_metrics.l1_cache_evictions_total.labels(reason="capacity").inc()
            pointer = self.allocator.allocate(size)

        transfer_start = time.monotonic()
        status = await self.transfer.copy_hbm_to_dram(hbm_addr, pointer.cpu_address, size)
        l1_metrics.l1_cache_transfer_duration_seconds.labels(direction="to_cpu").observe(
            time.monotonic() - transfer_start
        )

        if status.success:
            self._key_map[key] = pointer
            self.eviction_policy.track_new(key, size)
            l1_metrics.l1_cache_operations_total.labels(op="put", status="hit").inc()
            l1_metrics.l1_cache_transfer_bytes_total.labels(direction="to_cpu").inc(size)
            self._update_capacity_metrics()
            l1_metrics.l1_cache_operation_duration_seconds.labels(op="put").observe(
                time.monotonic() - start
            )
            return True

        self.allocator.free(pointer)
        l1_metrics.l1_cache_operations_total.labels(op="put", status="error").inc()
        return False

    async def get(self, key: str, dest_hbm_addr: int) -> bool:
        """Retrieve data from L1 to HBM."""
        start = time.monotonic()
        pointer = self._key_map.get(key)
        if not pointer:
            l1_metrics.l1_cache_operations_total.labels(op="get", status="miss").inc()
            return False

        transfer_start = time.monotonic()
        status = await self.transfer.copy_dram_to_hbm(pointer.cpu_address, dest_hbm_addr, pointer.size_bytes)
        l1_metrics.l1_cache_transfer_duration_seconds.labels(direction="to_gpu").observe(
            time.monotonic() - transfer_start
        )

        if status.success:
            self.eviction_policy.record_access(key)
            l1_metrics.l1_cache_operations_total.labels(op="get", status="hit").inc()
            l1_metrics.l1_cache_transfer_bytes_total.labels(direction="to_gpu").inc(pointer.size_bytes)
            l1_metrics.l1_cache_operation_duration_seconds.labels(op="get").observe(
                time.monotonic() - start
            )
            return True

        l1_metrics.l1_cache_operations_total.labels(op="get", status="error").inc()
        return False

    def has_key(self, key: str) -> bool:
        return key in self._key_map

    def _evict_victim(self, key: str) -> None:
        """Remove a specific key from L1."""
        pointer = self._key_map.pop(key, None)
        if pointer:
            self.allocator.free(pointer)
            self.eviction_policy.remove(key)
            self._update_capacity_metrics()
