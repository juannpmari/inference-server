"""L1 Cache memory allocator with free-list reclamation.

Manages a fixed-size pool of CPU DRAM. Tracks allocations with a sorted
free-list so that free() reclaims specific regions and subsequent allocations
can reuse them (first-fit strategy).
"""

import bisect
import logging
from typing import List, Optional, Tuple

from shared.types import AllocationPointer

logger = logging.getLogger(__name__)

# Base address for simulated memory pool
_POOL_BASE = 0x1000_0000


class L1Allocator:
    """Fixed-size CPU DRAM allocator with free-list reclamation."""

    def __init__(self, capacity_bytes: int, mock: bool = True):
        self._capacity = capacity_bytes
        self._mock = mock
        self._used = 0
        # Free-list: sorted list of (offset, size) tuples representing free regions
        self._free_list: List[Tuple[int, int]] = [(0, capacity_bytes)]
        # Active allocations: offset -> size
        self._allocations: dict[int, int] = {}
        logger.debug(f"L1 Allocator initialized: {capacity_bytes} bytes (mock={mock})")

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def available_bytes(self) -> int:
        return self._capacity - self._used

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def allocation_count(self) -> int:
        return len(self._allocations)

    def allocate(self, size_bytes: int) -> Optional[AllocationPointer]:
        """First-fit allocation from the free-list. Returns None if no space."""
        if size_bytes <= 0:
            return None

        for i, (offset, region_size) in enumerate(self._free_list):
            if region_size >= size_bytes:
                # Found a fit — remove or shrink the free region
                if region_size == size_bytes:
                    self._free_list.pop(i)
                else:
                    self._free_list[i] = (offset + size_bytes, region_size - size_bytes)

                self._used += size_bytes
                self._allocations[offset] = size_bytes
                address = _POOL_BASE + offset
                return AllocationPointer(cpu_address=address, size_bytes=size_bytes)

        return None  # No contiguous region large enough

    def free(self, ptr: AllocationPointer) -> bool:
        """Return an allocation to the free-list with coalescing."""
        offset = ptr.cpu_address - _POOL_BASE
        if offset not in self._allocations:
            logger.warning(f"Double-free or invalid pointer: {hex(ptr.cpu_address)}")
            return False

        size = self._allocations.pop(offset)
        self._used -= size

        # Insert into sorted free-list and coalesce adjacent regions
        new_region = (offset, size)
        insert_idx = bisect.bisect_left(self._free_list, new_region)
        self._free_list.insert(insert_idx, new_region)
        self._coalesce(insert_idx)
        return True

    def _coalesce(self, idx: int) -> None:
        """Merge adjacent free regions around the given index."""
        # Merge with right neighbor
        while idx + 1 < len(self._free_list):
            curr_offset, curr_size = self._free_list[idx]
            next_offset, next_size = self._free_list[idx + 1]
            if curr_offset + curr_size == next_offset:
                self._free_list[idx] = (curr_offset, curr_size + next_size)
                self._free_list.pop(idx + 1)
            else:
                break

        # Merge with left neighbor
        while idx > 0:
            prev_offset, prev_size = self._free_list[idx - 1]
            curr_offset, curr_size = self._free_list[idx]
            if prev_offset + prev_size == curr_offset:
                self._free_list[idx - 1] = (prev_offset, prev_size + curr_size)
                self._free_list.pop(idx)
                idx -= 1
            else:
                break
