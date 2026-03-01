"""gRPC service handler for KV Cache operations.

Receives requests from the engine and delegates to MultiTieredCacheManager.
"""

import logging
from typing import Any

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from shared.types import BlockReference, TransferResult

logger = logging.getLogger(__name__)


class KVCacheAPIService:
    """gRPC service exposing cache offload/fetch/query/eviction operations."""

    def __init__(self, cache_manager: MultiTieredCacheManager):
        self._manager = cache_manager
        logger.info("KV Cache API Service initialized")

    async def OffloadKVBlock(self, request: Any, context: Any) -> TransferResult:
        """Offload a KV block from GPU HBM to the cache hierarchy."""
        try:
            key = request.key
            block_ref = BlockReference(
                device_id=request.device_id,
                memory_address=request.memory_address,
                size_bytes=request.size_bytes,
            )
            model_id = getattr(request, "model_id", "")
            prefix_hash = getattr(request, "prefix_hash", "")
            return await self._manager.process_offload(
                block_ref, key, model_id=model_id, prefix_hash=prefix_hash
            )
        except Exception as e:
            return TransferResult(success=False, message=f"Offload failed: {e}")

    async def FetchKVBlock(self, request: Any, context: Any) -> TransferResult:
        """Fetch a KV block back into GPU HBM."""
        try:
            key = request.key
            dest_ref = BlockReference(
                device_id=request.dest_device_id,
                memory_address=request.dest_memory_address,
                size_bytes=request.dest_size_bytes,
            )
            return await self._manager.execute_fetch(key, dest_ref)
        except Exception as e:
            return TransferResult(success=False, message=f"Fetch failed: {e}")

    async def QueryAvailability(self, request: Any, context: Any) -> bool:
        """Check if a key exists in any cache tier."""
        return await self._manager.check_global_availability(request.key)

    async def RequestL1Space(self, request: Any, context: Any) -> TransferResult:
        """Force L1 eviction to free requested bytes."""
        return await self._manager.execute_l1_eviction(request.needed_size_bytes)
