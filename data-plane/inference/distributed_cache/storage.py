import asyncio
import logging
import redis.asyncio as redis
import time
from typing import Optional, Tuple, NamedTuple

# --- Configuration for this specific node ---
# NOTE: In a Kubernetes setup, these are configured for the specific pod.
NODE_ID = "redis-0" # Identifier used by the KVCache Watcher
REDIS_HOST = "localhost" # Assumes a Redis server is running locally or accessible
REDIS_PORT = 6379 
WATCHER_ADDR = "kv-cache-watcher:50051" # Address of the Controller

# --- Type Definitions ---
class StorageStatus(NamedTuple):
    success: bool
    message: str

class StorageNode:
    """
    Represents the actual L2 Storage Pod.
    Handles high-throughput PUT/GET requests from the L2 Connector.
    """
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.redis_client: Optional[redis.Redis] = None
        logging.info(f"Storage Node {node_id} configured for Redis at {host}:{port}")

    async def initialize(self):
        """Establishes the connection to the underlying Redis instance."""
        try:
            # Note: We assume Redis is running separately (e.g., another container)
            self.redis_client = redis.Redis(host=self.host, port=self.port, db=0)
            await self.redis_client.ping()
            logging.info("✅ Redis connection successful.")
        except Exception as e:
            logging.error(f"❌ Failed to connect to Redis: {e}")
            raise ConnectionError("Could not connect to Redis backend.")

    # --- Data Plane Interactions ---

    async def put_kv_block(self, key: str, data: bytes) -> StorageStatus:
        """
        Stores the raw, serialized KV Tensor data into the L2 cache.
        This is called by the L2 Connector.
        """
        if not self.redis_client:
            return StorageStatus(False, "Redis client not connected.")
        
        # Redis SET is fast and handles binary data
        try:
            await self.redis_client.set(key, data)
            logging.debug(f"PUT {key}: Stored {len(data)} bytes.")
            return StorageStatus(True, f"Stored {len(data)} bytes.")
        except Exception as e:
            return StorageStatus(False, f"Storage error: {str(e)}")

    async def get_kv_block(self, key: str) -> Tuple[StorageStatus, Optional[bytes]]:
        """
        Retrieves the KV Tensor data from the L2 cache.
        """
        if not self.redis_client:
            return StorageStatus(False, "Redis client not connected."), None
        
        try:
            data = await self.redis_client.get(key)
            if data is None:
                return StorageStatus(False, "Key not found."), None
            
            logging.debug(f"GET {key}: Retrieved {len(data)} bytes.")
            return StorageStatus(True, "Success"), data
        except Exception as e:
            return StorageStatus(False, f"Retrieval error: {str(e)}"), None

    # --- Control Plane (Heartbeat to Watcher) ---

    async def _send_heartbeat(self):
        """Simulates periodic reporting of health to the KVCache Watcher."""
        # NOTE: This would use a gRPC client to communicate with the KVCacheWatcher (Controller)
        # to confirm this node's health and availability.
        
        while True:
            await asyncio.sleep(3)
            current_time = int(time.time())
            # try:
            #     # Mock: Send gRPC Heartbeat(node_id=self.node_id, timestamp=current_time)
            #     logging.debug(f"Sending Heartbeat to Watcher...")
            # except Exception:
            #     logging.warning("Watcher unavailable. Health not reported.")
            pass # Keep loop clean for prototype

    async def run_node(self):
        """Starts the main runtime loops for the storage node."""
        await self.initialize()
        
        # Run the Heartbeat task concurrently
        heartbeat_task = asyncio.create_task(self._send_heartbeat())
        
        logging.info(f"Storage Node {self.node_id} is running and ready for data traffic.")
        # In a real scenario, this would block indefinitely waiting for Redis/InfiniStore traffic
        await asyncio.gather(heartbeat_task, return_exceptions=False)
        
        # Cleanup (if service is gracefully stopped)
        await self.redis_client.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    
    # Example: Run the node that the L2 Connector will target
    node = StorageNode(NODE_ID, REDIS_HOST, REDIS_PORT)
    try:
        asyncio.run(node.run_node())
    except ConnectionError:
        logging.error("Exiting due to critical connection failure.")
    except KeyboardInterrupt:
        logging.info("Node shut down by user.")