from typing import NamedTuple, Any, Optional
import asyncio
# Assuming the use of an asynchronous gRPC server library (e.g., aiogrpc or native asyncio)

# --- Type Definitions for the API ---

class BlockReference(NamedTuple):
    """Reference to a specific KV block in GPU or CPU memory."""
    device_id: int
    memory_address: int
    size_bytes: int

class Status(NamedTuple):
    """Simple status object for operation results."""
    success: bool
    message: str

# --- Abstract Dependencies ---

class MultiTieredCacheManager:
    """
    Conceptual interface for the strategic 'brain' of the sidecar.
    This class handles the logic, policy, and directs the Connector.
    """
    def __init__(self, connector: Any):
        self._connector = connector
        
    async def process_offload(self, block_ref: BlockReference) -> Status:
        """Decides where to offload the block (L1 or L2) and initiates the move."""
        raise NotImplementedError

    async def execute_fetch(self, key: str, dest_ref: BlockReference) -> Status:
        """Looks up the key in L1, then L2, and initiates the transfer to the destination."""
        raise NotImplementedError

    async def check_global_availability(self, key: str) -> bool:
        """Queries the L1 and L2 Meta Service for the key's presence."""
        raise NotImplementedError

    async def execute_l1_eviction(self, needed_size: int) -> Status:
        """Triggers the L1 eviction policy to free up space in CPU DRAM."""
        raise NotImplementedError


