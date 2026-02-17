from typing import Optional, Dict
from allocator import L1Allocator, AllocationPointer
from gpu_transfer import GPUTransferHandler
from eviction_policies import LRUPolicy

class L1CacheAPI:
    """
    The L1 Facade. 
    It coordinates the Allocator, Transfer Handler, and Eviction Policy 
    to provide simple 'put' and 'get' operations for the Cache Manager.
    """
    def __init__(self, capacity_gb: float = 1.0):
        # 1. Initialize Components
        capacity_bytes = int(capacity_gb * 1024**3)
        self.allocator = L1Allocator(capacity_bytes)
        self.transfer = GPUTransferHandler()
        self.eviction_policy = LRUPolicy()
        
        # Internal map: Key -> AllocationPointer
        # Used to find the physical address when given a logical key
        self._key_map: Dict[str, AllocationPointer] = {}

    async def put(self, key: str, hbm_addr: int, size: int) -> bool:
        """
        Offloads data from HBM to L1. Handles allocation and automatic eviction.
        """
        # 1. Try Allocate
        pointer = self.allocator.allocate(size)

        # 2. If full, run Eviction Loop
        while pointer is None:
            victim_key = self.eviction_policy.select_victim()
            if not victim_key: 
                print("Error: L1 Full and no victims to evict!")
                return False
            
            # Evict the victim
            print(f"Evicting L1 victim: {victim_key}")
            self._evict_victim(victim_key)
            
            # Retry allocation
            pointer = self.allocator.allocate(size)

        # 3. Perform Transfer
        status = await self.transfer.copy_hbm_to_dram(hbm_addr, pointer.cpu_address, size)
        
        if status.success:
            # 4. Update State
            self._key_map[key] = pointer
            self.eviction_policy.track_new(key, size)
            return True
        else:
            # Rollback allocation if transfer failed
            self.allocator.free(pointer)
            return False

    async def get(self, key: str, dest_hbm_addr: int) -> bool:
        """
        Retrieves data from L1 to HBM.
        """
        pointer = self._key_map.get(key)
        if not pointer:
            return False  # Cache Miss

        # 1. Transfer
        status = await self.transfer.copy_dram_to_hbm(pointer.cpu_address, dest_hbm_addr, pointer.size_bytes)
        
        # 2. Update Policy (LRU touch)
        if status.success:
            self.eviction_policy.record_access(key)
            return True
        return False

    def _evict_victim(self, key: str):
        """Internal helper to remove a specific key from L1."""
        pointer = self._key_map.pop(key, None)
        if pointer:
            self.allocator.free(pointer)
            self.eviction_policy.remove(key)