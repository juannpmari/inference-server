"""Shared type definitions used across the inference server.

Consolidates duplicate NamedTuples previously defined independently in
cache_manager.py, gpu_transfer.py, connector.py, controller.py, and storage.py.
"""

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional


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


@dataclass
class KVBlockEntry:
    """Metadata for a cached KV block tracked by the KVBlockRegistry."""
    key: str
    location: Literal["L1", "L2"]
    size_bytes: int
    l1_address: Optional[int] = None
    l2_node_id: Optional[str] = None
    model_id: str = ""
    prefix_hash: str = ""
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "location": self.location,
            "size_bytes": self.size_bytes,
            "l1_address": self.l1_address,
            "l2_node_id": self.l2_node_id,
            "model_id": self.model_id,
            "prefix_hash": self.prefix_hash,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KVBlockEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
