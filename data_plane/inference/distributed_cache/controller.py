import asyncio
import logging
import grpc
from typing import List, Dict, NamedTuple
import random

# --- Configuration ---
CONTROLLER_PORT = 50051

# --- Type Definitions ---
class StorageNode(NamedTuple):
    node_id: str
    host: str
    port: int
    health: str # "UP", "DOWN"

# --- Mock gRPC Imports ---
# You would use generated code here (e.g., from kv_watcher_pb2_grpc)
class MockClusterMapRequest:
    pass

class MockClusterMapResponse:
    """Simulates the protobuf response containing a list of active nodes."""
    def __init__(self, nodes: List[StorageNode]):
        self.nodes = nodes

# --- The Central Control Plane ---

class KVCacheWatcher(kv_cache_pb2_grpc.KVCacheWatcherServicer):
    """
    The Central Controller (Meta Service) running on a dedicated pod.
    Manages topology, health, and routing metadata for the L2 Storage Cluster.
    """
    def __init__(self):
        # Maps node_id to StorageNode object
        self._topology: Dict[str, StorageNode] = {}
        # Simple placeholder for key ownership mapping (for complex sharding)
        self._key_map: Dict[str, str] = {}
        self._initialize_mock_nodes()
        logging.info("Watcher initialized with initial storage nodes.")

    def _initialize_mock_nodes(self):
        """Simulate initial Redis/InfiniStore pods."""
        self._topology["redis-0"] = StorageNode("redis-0", "10.1.1.10", 6379, "UP")
        self._topology["redis-1"] = StorageNode("redis-1", "10.1.1.11", 6379, "UP")
        self._topology["redis-2"] = StorageNode("redis-2", "10.1.1.12", 6379, "UP")

    def _get_active_nodes(self) -> List[StorageNode]:
        """Returns a list of nodes currently marked as UP."""
        return [node for node in self._topology.values() if node.health == "UP"]

    # --- gRPC Service Methods ---

    async def GetClusterMap(self, request: MockClusterMapRequest, context) -> MockClusterMapResponse:
        """
        [CONTROL PATH] Method called by the L2 Connector to get the current cluster topology.
        This provides the addresses needed for client-side routing (Consistent Hashing).
        """
        active_nodes = self._get_active_nodes()
        logging.info(f"Serving ClusterMap request. Returning {len(active_nodes)} active nodes.")
        return MockClusterMapResponse(nodes=active_nodes)

    async def Heartbeat(self, request, context) -> kv_cache_pb2.HeartbeatResponse:
        """
        [CONTROL PATH] Method called by Storage Nodes to report health.
        """
        # In a real system, this updates self._topology health status
        logging.debug(f"Received Heartbeat from {request.node_id}")
        return kv_cache_pb2.HeartbeatResponse(status="ACK")

    # --- Internal Management Logic (Simulation) ---

    async def _run_health_check_loop(self):
        """Simulates an internal loop checking node health (e.g., external pings)."""
        while True:
            await asyncio.sleep(5) # Check every 5 seconds
            
            # Simulate a node failure occasionally
            if random.random() < 0.05:
                node_id = random.choice(list(self._topology.keys()))
                old_health = self._topology[node_id].health
                new_health = "DOWN" if old_health == "UP" else "UP"
                
                # Update the topology
                self._topology[node_id] = self._topology[node_id]._replace(health=new_health)
                logging.warning(f"Simulated Health Change: Node {node_id} is now {new_health}!")
                
    # --- Other Controller Functions (Not implemented here) ---
    # - Key Migration Logic (for rebalancing)
    # - Data Shard Mapping updates

async def serve():
    """Starts the KVCache Watcher gRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [Watcher] - %(levelname)s - %(message)s'
    )
    
    server = grpc.aio.server()
    watcher_service = KVCacheWatcher()
    
    # kv_cache_pb2_grpc.add_KVCacheWatcherServicer_to_server(watcher_service, server)
    
    listen_addr = f'[::]:{CONTROLLER_PORT}'
    server.add_insecure_port(listen_addr)
    logging.info(f"🚀 KVCache Watcher starting on {listen_addr}")
    
    server.start()
    
    # Run the internal health check loop concurrently
    health_task = asyncio.create_task(watcher_service._run_health_check_loop())
    
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        pass
    finally:
        health_task.cancel()
        await server.stop(grace=5)

if __name__ == '__main__':
    # NOTE: You must have your gRPC stubs generated to run this.
    # We run the mock service for demonstration.
    asyncio.run(serve())