import asyncio
import logging
import grpc

# --- Import our custom modules ---
from kv_cache_api import KVCacheAPIService, MultiTieredCacheManager
from l1_cache_api import L1CacheAPI
from connector import L2Connector

# --- Import generated gRPC code ---
# Note: In a real project, you must generate these from your .proto file
# using: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. kv_cache.proto
import kv_cache_pb2
import kv_cache_pb2_grpc

# --- Configuration ---
GRPC_PORT = 50051
L1_CAPACITY_GB = 2.0  # Example: 2GB of CPU DRAM for L1

async def serve():
    """
    Main entry point for the Sidecar Application.
    """
    # 1. Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [Sidecar] - %(levelname)s - %(message)s'
    )
    logging.info("Starting AIBrix-style Offloading Connector Sidecar...")

    # 2. Initialize the L2 Connector (Remote Tier)
    # This establishes connections to the KVCache Watcher and Redis Storage
    l2_connector = L2Connector()
    await l2_connector.initialize()
    logging.info("✅ L2 Connector initialized (Connected to Watcher & Storage).")

    # 3. Initialize the L1 Cache API (Local Tier)
    # This allocates the CPU DRAM pool and sets up the eviction policy
    l1_api = L1CacheAPI(capacity_gb=L1_CAPACITY_GB)
    logging.info(f"✅ L1 Cache API initialized with {L1_CAPACITY_GB} GB capacity.")

    # 4. Initialize the Multi-Tiered Cache Manager (The Brain)
    # We inject both the L1 API and L2 Connector into the manager.
    # The manager doesn't need to know 'how' they work, just that they exist.
    cache_manager = MultiTieredCacheManager(
        l1_api=l1_api,
        l2_connector=l2_connector
    )
    logging.info("✅ Multi-Tiered Cache Manager initialized.")

    # 5. Initialize the gRPC Service
    # This is the interface the Inference Engine will call.
    server = grpc.aio.server()
    kv_service = KVCacheAPIService(cache_manager)
    
    # Register the service with the gRPC server
    kv_cache_pb2_grpc.add_KVCacheServiceServicer_to_server(kv_service, server)
    
    # 6. Bind and Start the Server
    listen_addr = f'[::]:{GRPC_PORT}'
    server.add_insecure_port(listen_addr)
    logging.info(f"🚀 Sidecar gRPC Server listening on {listen_addr}")
    
    await server.start()
    
    # 7. Keep the application running
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        logging.info("Shutting down sidecar...")
    finally:
        # Graceful cleanup
        await l2_connector.close()
        await server.stop(grace=5)

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass