"""Sidecar offloading backend — tracks block IDs for the sidecar tier.

Implements vLLM's Backend / BlockStatus / LoadStoreSpec interfaces.
Pure arithmetic — no gRPC calls. The handler does the actual I/O.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from vllm.v1.kv_offload.backend import Backend, BlockStatus
    from vllm.v1.kv_offload.abstract import BlockIDsLoadStoreSpec
    VLLM_OFFLOAD_AVAILABLE = True
except ImportError:
    VLLM_OFFLOAD_AVAILABLE = False

    class Backend:
        def __init__(self, block_size: int, medium: str = ""):
            pass

    class BlockStatus:
        def __init__(self):
            self.ref_cnt = -1

        @property
        def is_ready(self) -> bool:
            return self.ref_cnt >= 0

    class BlockIDsLoadStoreSpec:
        block_ids: np.ndarray
        @staticmethod
        def medium() -> str:
            return ""


class SidecarLoadStoreSpec(BlockIDsLoadStoreSpec):
    """Describes a set of blocks to store/load to/from sidecar."""

    def __init__(self, block_ids: np.ndarray, block_hashes: list[str]):
        self.block_ids = block_ids
        self.block_hashes = block_hashes

    @staticmethod
    def medium() -> str:
        return "SIDECAR"


class SidecarBackend(Backend):
    """Tracks block IDs for the sidecar tier. Same pattern as CPUBackend.

    Manages a local free-list of integer block IDs. No network calls.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        super().__init__(block_size=block_size, medium="SIDECAR")
        self._num_blocks = num_blocks
        self._free_ids: list[int] = list(range(num_blocks))
        self._allocated: dict[int, str] = {}  # block_id -> block_hash
        self._hash_to_id: dict[str, int] = {}
        self._hash_to_status: dict[str, BlockStatus] = {}
        self._status_to_id: dict[int, int] = {}  # id(BlockStatus) -> block_id
        logger.info(f"SidecarBackend initialized with {num_blocks} blocks")

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def get_num_free_blocks(self) -> int:
        return len(self._free_ids)

    def allocate_blocks(self, block_hashes: list[str]) -> list[BlockStatus]:
        """Assign integer IDs from free-list for the given hashes.

        Returns list of BlockStatus instances (ref_cnt=-1 means not yet ready).
        Caller must check get_num_free_blocks() beforehand.
        """
        statuses = []
        for bh in block_hashes:
            # If already allocated (e.g. duplicate hash), return existing
            if bh in self._hash_to_status:
                statuses.append(self._hash_to_status[bh])
                continue
            block_id = self._free_ids.pop()
            self._allocated[block_id] = bh
            self._hash_to_id[bh] = block_id

            status = BlockStatus()
            # ref_cnt=-1 means "not ready"; vLLM's complete_store sets it to 0
            self._hash_to_status[bh] = status
            self._status_to_id[id(status)] = block_id
            statuses.append(status)
        return statuses

    def free(self, block: BlockStatus) -> bool:
        """Free a block by its BlockStatus instance."""
        block_id = self._status_to_id.pop(id(block), None)
        if block_id is None:
            return False
        bh = self._allocated.pop(block_id, None)
        if bh is not None:
            self._hash_to_id.pop(bh, None)
            self._hash_to_status.pop(bh, None)
        self._free_ids.append(block_id)
        return True

    def get_block_id(self, block: BlockStatus) -> Optional[int]:
        """Look up the integer block ID for a BlockStatus."""
        return self._status_to_id.get(id(block))

    def get_load_store_spec(
        self,
        block_hashes,
        blocks,
    ) -> SidecarLoadStoreSpec:
        """Return a spec describing which blocks to transfer.

        Args:
            block_hashes: iterable of BlockHash identifying the blocks.
            blocks: iterable of BlockStatus returned by allocate_blocks.
        """
        block_hashes = list(block_hashes)
        block_ids = [self._status_to_id[id(b)] for b in blocks]
        return SidecarLoadStoreSpec(
            block_ids=np.array(block_ids, dtype=np.int64),
            block_hashes=block_hashes,
        )
