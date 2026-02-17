from typing import Optional, NamedTuple

class AllocationPointer(NamedTuple):
    """Represents a successful allocation in CPU DRAM."""
    cpu_address: int  # The physical/virtual memory address
    size_bytes: int

class L1Allocator:
    """
    Manages a fixed-size pool of pinned CPU DRAM for L1 caching.
    Focuses solely on memory math and reservation.
    """
    def __init__(self, capacity_bytes: int):
        self._capacity = capacity_bytes
        self._used = 0
        # In a real implementation, you would perform the actual cudaHostAlloc here.
        print(f"DEBUG: L1 Allocator initialized with {capacity_bytes} bytes.")

    @property
    def available_space(self) -> int:
        return self._capacity - self._used

    def allocate(self, size_bytes: int) -> Optional[AllocationPointer]:
        """
        Attempts to reserve space. Returns a pointer if successful, None if full.
        """
        if self._used + size_bytes > self._capacity:
            return None
        
        # Simulate memory address offset
        address = 0x10000000 + self._used
        self._used += size_bytes
        return AllocationPointer(cpu_address=address, size_bytes=size_bytes)

    def free(self, ptr: AllocationPointer) -> None:
        """Releases the memory space back to the pool."""
        self._used -= ptr.size_bytes
        # Real implementation would manage fragmentation/freelist here.
        print(f"DEBUG: Freed {ptr.size_bytes} bytes at {hex(ptr.cpu_address)}")