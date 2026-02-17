"""Shared type definitions used across the inference server.

Consolidates duplicate NamedTuples previously defined independently in
cache_manager.py, gpu_transfer.py, connector.py, controller.py, and storage.py.
"""

from typing import NamedTuple, Optional


class BlockReference(NamedTuple):
    """Reference to a KV cache block in GPU HBM."""
    device_id: int
    memory_address: int
    size_bytes: int


class TransferResult(NamedTuple):
    """Outcome of a data transfer or cache operation."""
    success: bool
    message: str
    data: Optional[bytes] = None


class StorageNodeInfo(NamedTuple):
    """Describes an L2 storage node (Redis pod) in the cluster."""
    node_id: str
    host: str
    port: int
    health: str = "UP"


class AllocationPointer(NamedTuple):
    """Represents a successful allocation in CPU DRAM."""
    cpu_address: int
    size_bytes: int
