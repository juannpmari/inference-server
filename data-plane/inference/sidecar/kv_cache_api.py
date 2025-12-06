from inference.sidecar.cache_manager import MultiTieredCacheManager

class KVCacheAPIService:
    """
    The gRPC service handler for KV Cache operations. 
    This is the 'KV Cache API' exposed by the Offloading Connector sidecar.
    It receives requests and delegates all strategic decisions to the Cache Manager.
    """

    def __init__(self, cache_manager: MultiTieredCacheManager):
        """
        Initializes the service with a reference to the core logic module.
        
        Args:
            cache_manager: The instance of the MultiTieredCacheManager responsible for logic.
        """
        self._manager = cache_manager
        print("KV Cache API Service initialized and ready to receive gRPC requests.")

    async def OffloadKVBlock(self, request: Any, context: Any) -> Status:
        """
        Handles the Engine's request to offload a KV block.
        
        This method receives a BlockReference and immediately passes control to the 
        Cache Manager, which decides whether to move the data to L1 or L2.
        
        Args:
            request: The gRPC request object containing BlockReference data.
            context: The gRPC context object.
            
        Returns:
            Status: An object indicating the success or failure of the offload process.
        """
        try:
            # Assume request fields map to BlockReference
            block_ref = BlockReference(
                device_id=request.device_id,
                memory_address=request.memory_address,
                size_bytes=request.size_bytes
            )
            # Delegate strategy to the manager
            result_status = await self._manager.process_offload(block_ref)
            return result_status
        except Exception as e:
            # Handle potential gRPC or internal errors
            return Status(success=False, message=f"Offload failed: {str(e)}")


    # --- gRPC Method 2: Fetch ---
    async def FetchKVBlock(self, request: Any, context: Any) -> Status:
        """
        Handles the Engine's request to fetch KV data for a specific key.
        
        The manager will perform a multi-tiered lookup (L1 -> L2) and ensure the 
        data is placed back into the HBM address provided by the destination reference.
        
        Args:
            request: The gRPC request object containing the key and destination reference.
            context: The gRPC context object.
            
        Returns:
            Status: An object indicating the success or failure of the fetch process.
        """
        try:
            key = request.key  # The unique prefix hash
            dest_ref = BlockReference(
                device_id=request.dest_device_id,
                memory_address=request.dest_memory_address,
                size_bytes=request.dest_size_bytes
            )
            # Delegate lookup and transfer execution to the manager
            result_status = await self._manager.execute_fetch(key, dest_ref)
            return result_status
        except Exception as e:
            return Status(success=False, message=f"Fetch failed: {str(e)}")


    # --- gRPC Method 3: Query (Optional but useful) ---
    async def QueryAvailability(self, request: Any, context: Any) -> bool:
        """
        Allows the Engine to quickly check if a specific key is available in the 
        L1 or L2 cache without initiating a transfer.
        
        Args:
            request: The gRPC request object containing the key.
            context: The gRPC context object.
            
        Returns:
            bool: True if the key is found in any tier, False otherwise.
        """
        key = request.key
        is_available = await self._manager.check_global_availability(key)
        return is_available


    # --- gRPC Method 4: L1 Eviction/Space Request ---
    async def RequestL1Space(self, request: Any, context: Any) -> Status:
        """
        Called by the Engine's scheduler when it anticipates running out of L1 (DRAM) space.
        This forces the Cache Manager to run the L1 eviction policy immediately.
        
        Args:
            request: The gRPC request object containing the needed size in bytes.
            context: The gRPC context object.
            
        Returns:
            Status: Indicating whether the required space was freed.
        """
        needed_size = request.needed_size_bytes
        result_status = await self._manager.execute_l1_eviction(needed_size)
        return result_status