"""Block slot allocator for L1 byte store.

Manages a fixed number of block slots identified by integer IDs.
No memory addresses, no coalescing — just a free-list of block IDs.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BlockSlotAllocator:
    """Fixed-capacity allocator tracking block IDs with a free-list."""

    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self._free_ids: list[int] = list(range(num_blocks))
        self._allocated: set[int] = set()
        logger.debug(f"BlockSlotAllocator initialized: {num_blocks} slots")

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    @property
    def num_allocated(self) -> int:
        return len(self._allocated)

    def allocate(self) -> Optional[int]:
        """Allocate a single block slot. Returns block_id or None if full."""
        if not self._free_ids:
            return None
        block_id = self._free_ids.pop()
        self._allocated.add(block_id)
        return block_id

    def allocate_n(self, n: int) -> Optional[list[int]]:
        """Allocate n block slots. Returns list of IDs or None if not enough free."""
        if len(self._free_ids) < n:
            return None
        ids = [self._free_ids.pop() for _ in range(n)]
        self._allocated.update(ids)
        return ids

    def free(self, block_id: int) -> bool:
        """Return a block slot to the free-list."""
        if block_id not in self._allocated:
            logger.warning(f"Double-free or invalid block_id: {block_id}")
            return False
        self._allocated.discard(block_id)
        self._free_ids.append(block_id)
        return True

    def is_allocated(self, block_id: int) -> bool:
        return block_id in self._allocated
