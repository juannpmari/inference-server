import asyncio
import logging
import hashlib
import grpc
import redis.asyncio as redis
from typing import List, Optional, Dict, Tuple, NamedTuple

# --- Configuration ---
# In a real K8s setup, these would be env vars
WATCHER_SERVICE_ADDR = "kv-cache-watcher:50051"
REDIS_PORT = 6379

# Mock Protobuf imports (You would generate these from .proto files)
# from kv_watcher_pb2 import ClusterMapRequest
# from kv_watcher_pb2_grpc import KVCacheWatcherStub

class StorageNode(NamedTuple):
    node_id: str
    host: str
    port: int

class TransferStatus(NamedTuple):
    success: bool
    message: str
    data: Optional[bytes] = None

# --- 1. Placement Policy (Consistent Hashing) ---

class ConsistentHashRing:
    """
    Determines which Storage Node holds a specific Key.
    This replicates the 'L2 Placement Policy' logic locally in the sidecar.
    """
    def __init__(self, replicas: int = 3):
        self.replicas = replicas
        self.ring: Dict[int, StorageNode] = {}
        self.sorted_keys: List[int] = []

    def _hash(self, key: str) -> int:
        """SHA-256 hash for stable distribution."""
        return int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16)

    def update_nodes(self, nodes: List[StorageNode]):
        """Rebuilds the hash ring with the new list of active nodes."""
        self.ring.clear()
        self.sorted_keys.clear()
        
        for node in nodes:
            for i in range(self.replicas):
                # Create virtual nodes to ensure better distribution
                virtual_node_key = f"{node.node_id}-{i}"
                key_hash = self._hash(virtual_node_key)
                self.ring[key_hash] = node
                self.sorted_keys.append(key_hash)
        
        self.sorted_keys.sort()
        logging.info(f"Updated Hash Ring with {len(nodes)} nodes.")

    def get_node(self, key: str) -> Optional[StorageNode]:
        """Finds the node responsible for the given key."""
        if not self.ring:
            return None
        
        key_hash = self._hash(key)
        # Binary search for the first node hash >= key_hash
        # (This is a simplified linear scan for clarity; bisect is faster)
        for node_hash in self.sorted_keys:
            if key_hash <= node_hash:
                return self.ring[node_hash]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]

# --- 2. Control Plane Client (Meta Service) ---

class KVCacheWatcherClient:
    """
    Communicates with the Centralized Cluster Manager (Watcher) via gRPC.
    """
    def __init__(self, address: str):
        self.address = address
        self.channel = None
        self.stub = None

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(self.address)
        # self.stub = KVCacheWatcherStub(self.channel)
        logging.info(f"Connected to KV Cache Watcher at {self.address}")

    async def get_active_nodes(self) -> List[StorageNode]:
        """
        Asks the Watcher for the current list of healthy Redis pods.
        """
        # In a real app, you would await self.stub.GetClusterMap(ClusterMapRequest())
        # Here we mock the response for the replica:
        await asyncio.sleep(0.05) # Simulate network RTT
        
        # MOCK: Returning 2 Redis pods
        return [
            StorageNode("redis-0", "127.0.0.1", 6379),
            StorageNode("redis-1", "127.0.0.1", 6380) # Using different port for simulation
        ]

# --- 3. The L2 Connector (Main Orchestrator) ---

class L2Connector:
    """
    The L2 Facade.
    Coordinates the Watcher (for topology) and Redis Clients (for data).
    """
    def __init__(self):
        self.watcher = KVCacheWatcherClient(WATCHER_SERVICE_ADDR)
        self.hasher = ConsistentHashRing()
        # Pool of active Redis clients: { "host:port": RedisClient }
        self.redis_pool: Dict[str, redis.Redis] = {}

    async def initialize(self):
        """Starts the connector and syncs the initial cluster map."""
        await self.watcher.connect()
        await self._refresh_cluster_map()
        logging.info("L2 Connector Initialized.")

    async def _refresh_cluster_map(self):
        """Fetches new topology and updates the hash ring."""
        nodes = await self.watcher.get_active_nodes()
        self.hasher.update_nodes(nodes)

    async def _get_redis_client(self, node: StorageNode) -> redis.Redis:
        """Gets or creates an async Redis client for the specific node."""
        node_key = f"{node.host}:{node.port}"
        if node_key not in self.redis_pool:
            # Create a new async connection
            client = redis.Redis(host=node.host, port=node.port, db=0)
            self.redis_pool[node_key] = client
        return self.redis_pool[node_key]

    async def put(self, key: str, data: bytes) -> TransferStatus:
        """
        L2 WRITE:
        1. Find correct node (Placement Policy).
        2. Serialize (Already bytes).
        3. Send to Redis.
        """
        node = self.hasher.get_node(key)
        if not node:
            return TransferStatus(False, "No storage nodes available.")

        try:
            client = await self._get_redis_client(node)
            # Use SET with simple binary safety
            await client.set(key, data)
            return TransferStatus(True, f"Stored on {node.node_id}")
        except Exception as e:
            return TransferStatus(False, f"Redis error on {node.node_id}: {str(e)}")

    async def get(self, key: str) -> TransferStatus:
        """
        L2 READ:
        1. Find correct node (Placement Policy).
        2. Fetch from Redis.
        """
        node = self.hasher.get_node(key)
        if not node:
            return TransferStatus(False, "No storage nodes available.")

        try:
            client = await self._get_redis_client(node)
            data = await client.get(key)
            
            if data is None:
                return TransferStatus(False, "Key not found (Cache Miss)")
            
            return TransferStatus(True, "Success", data)
        except Exception as e:
            return TransferStatus(False, f"Redis error on {node.node_id}: {str(e)}")

    async def close(self):
        """Cleanup connections."""
        for client in self.redis_pool.values():
            await client.aclose()
        if self.watcher.channel:
            await self.watcher.channel.close()