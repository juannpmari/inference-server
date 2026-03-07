"""Sidecar offloading backend — tracks block IDs for the sidecar tier.

Implements vLLM's Backend / BlockStatus / LoadStoreSpec interfaces.
Pure arithmetic — no gRPC calls. The handler does the actual I/O.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        Backend,
        BlockStatus,
        LoadStoreSpec,
    )
    VLLM_OFFLOAD_AVAILABLE = True
except ImportError:
    VLLM_OFFLOAD_AVAILABLE = False

    class Backend:
        pass

    class BlockStatus:
        pass

    class LoadStoreSpec:
        pass


class SidecarBlockStatus(BlockStatus):
    """Status of a block in the sidecar."""

    def __init__(self, block_id: int, stored: bool = False):
        self.block_id = block_id
        self.stored = stored


@dataclass
class SidecarLoadStoreSpec(LoadStoreSpec):
    """Describes a set of blocks to store/load to/from sidecar."""

    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)

    def medium(self) -> str:
        return "SIDECAR"


class SidecarBackend(Backend):
    """Tracks block IDs for the sidecar tier. Same pattern as CPUBackend.

    Manages a local free-list of integer block IDs. No network calls.
    """

    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self._free_ids: list[int] = list(range(num_blocks))
        self._allocated: dict[int, str] = {}  # block_id -> block_hash
        self._hash_to_id: dict[str, int] = {}
        logger.info(f"SidecarBackend initialized with {num_blocks} blocks")

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def get_num_free_blocks(self) -> int:
        return len(self._free_ids)

    def allocate_blocks(self, block_hashes: list[str]) -> Optional[list[int]]:
        """Assign integer IDs from free-list for the given hashes."""
        if len(self._free_ids) < len(block_hashes):
            return None
        ids = []
        for bh in block_hashes:
            # If already allocated (e.g. duplicate hash), return existing
            if bh in self._hash_to_id:
                ids.append(self._hash_to_id[bh])
                continue
            block_id = self._free_ids.pop()
            self._allocated[block_id] = bh
            self._hash_to_id[bh] = block_id
            ids.append(block_id)
        return ids

    def free_block(self, block_id: int) -> bool:
        bh = self._allocated.pop(block_id, None)
        if bh is None:
            return False
        self._hash_to_id.pop(bh, None)
        self._free_ids.append(block_id)
        return True

    def get_load_store_spec(
        self,
        block_ids: list[int],
        block_hashes: list[str],
    ) -> SidecarLoadStoreSpec:
        """Return a spec describing which blocks to transfer."""
        return SidecarLoadStoreSpec(
            block_ids=block_ids,
            block_hashes=block_hashes,
        )
