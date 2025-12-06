import logging
from typing import NamedTuple
from l1_cache_api import L1CacheAPI
from connector import L2Connector

# Re-using definitions from previous steps
class BlockReference(NamedTuple):
    device_id: int
    memory_address: int
    size_bytes: int

class Status(NamedTuple):
    success: bool
    message: str

class MultiTieredCacheManager:
    """
    The 'Brain' of the Sidecar.
    Implements the Dynamic Balancing Logic between L1 and L2.
    """
    def __init__(self, l1_api: L1CacheAPI, l2_connector: L2Connector):
        self.l1 = l1_api
        self.l2 = l2_connector

    async def process_offload(self, block_ref: BlockReference, key: str) -> Status:
        """
        STRATEGY:
        1. Try to put in L1 (Fastest).
        2. If L1 is full/fails, promote to L2 (Persistence/Sharing).
        """
        logging.debug(f"Processing offload for key: {key}")

        # Tier 1: Try L1 Offload (Local RAM)
        # Note: L1 API handles the HBM -> DRAM transfer internally
        l1_success = await self.l1.put(key, block_ref.memory_address, block_ref.size_bytes)
        
        if l1_success:
            return Status(True, "Offloaded to L1")

        # Tier 2: L1 failed (Full & Eviction didn't help), so go to L2 (Remote Redis)
        logging.info(f"L1 Full/Failed for {key}. Promoting to L2...")
        
        # For L2, we first need to pull the data from GPU to a temporary buffer
        # (In a real implementation, you might stream this directly)
        # For this prototype, we assume we have a way to get the bytes:
        # data_bytes = await gpu_transfer.read_gpu_memory(block_ref) <--- Conceptual helper
        
        # MOCKING the data read for the prototype
        mock_data = b'\x00' * block_ref.size_bytes 
        
        l2_status = await self.l2.put(key, mock_data)
        return Status(l2_status.success, f"Promoted to L2: {l2_status.message}")

    async def execute_fetch(self, key: str, dest_ref: BlockReference) -> Status:
        """
        STRATEGY:
        1. Check L1.
        2. If Miss, Check L2.
        """
        # Tier 1: Check L1
        l1_success = await self.l1.get(key, dest_ref.memory_address)
        if l1_success:
            return Status(True, "Hit in L1")

        # Tier 2: Check L2
        l2_status = await self.l2.get(key)
        if l2_status.success and l2_status.data:
            # We got the data bytes from Redis. Now we must load them into GPU HBM.
            # We use the L1 API's transfer handler (or a dedicated one) to write to HBM.
            # await gpu_transfer.write_to_hbm(dest_ref.memory_address, l2_status.data)
            return Status(True, "Hit in L2 (Restored to HBM)")
        
        return Status(False, "Cache Miss (Key not found in L1 or L2)")

    async def check_global_availability(self, key: str) -> bool:
        """Queries the L1 and L2 Meta Service for the key's presence."""
        raise NotImplementedError

    async def execute_l1_eviction(self, needed_size: int) -> Status:
        """Triggers the L1 eviction policy to free up space in CPU DRAM."""
        raise NotImplementedError


