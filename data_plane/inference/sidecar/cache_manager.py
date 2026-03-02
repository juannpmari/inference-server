"""Multi-tiered cache manager orchestrating L1 byte store and L2.

Accepts raw bytes (not GPU addresses). The engine serializes tensors to bytes,
sends them over gRPC, and this manager stores them in L1 (fast, local) with
optional L2 fallback (slower, distributed Redis).
"""

import logging
import time
from typing import Optional

from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from data_plane.inference.sidecar.l1_cache.api import L1ByteStore
from data_plane.inference.sidecar.l2_cache.connector import L2Connector
from shared.types import KVBlockEntry

logger = logging.getLogger(__name__)


class MultiTieredCacheManager:
    """Orchestrates L1 byte store + L2 with registry tracking."""

    def __init__(
        self,
        l1: L1ByteStore,
        l2: L2Connector,
        registry: Optional[KVBlockRegistry] = None,
    ):
        self.l1 = l1
        self.l2 = l2
        self.registry = registry or KVBlockRegistry()

    def get_num_free_blocks(self) -> int:
        return self.l1.get_num_free_blocks()

    def allocate_blocks(self, block_hashes: list[str]) -> Optional[list[int]]:
        """Allocate block slots in L1. Returns block IDs or None if insufficient space."""
        return self.l1.allocate_blocks(block_hashes)

    async def store_block(
        self,
        block_id: int,
        block_hash: str,
        data: bytes,
        model_id: str = "",
        layer_name: str = "",
    ) -> bool:
        """Store block bytes in L1, register in metadata."""
        ok = self.l1.store(block_id, data, block_hash)
        if ok:
            self.registry.register(KVBlockEntry(
                key=block_hash,
                location="L1",
                size_bytes=len(data),
                model_id=model_id,
                prefix_hash=block_hash,
                created_at=time.time(),
                last_accessed=time.time(),
            ))
        return ok

    async def load_block(self, block_id: int) -> Optional[bytes]:
        """Load block bytes from L1. Returns None on miss."""
        data = self.l1.load(block_id)
        if data is not None:
            block_hash = self.l1._id_to_hash.get(block_id)
            if block_hash:
                self.registry.record_access(block_hash)
        return data

    def free_block(self, block_id: int) -> bool:
        """Free a block slot and unregister from metadata."""
        block_hash = self.l1._id_to_hash.get(block_id)
        result = self.l1.free(block_id)
        if result and block_hash:
            self.registry.unregister(block_hash)
        return result
