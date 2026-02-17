"""Protocol interfaces for testability.

Three targeted ports that allow swapping real implementations for fakes in tests:
- CacheStore: abstracts L1 and L2 cache backends
- ModelRepository: abstracts model/adapter artifact fetching
- GPUTransfer: abstracts CUDA memory copy operations
"""

from typing import Optional, Protocol, runtime_checkable

from shared.types import TransferResult


@runtime_checkable
class CacheStore(Protocol):
    """Unified interface for any key-value cache tier (L1 DRAM or L2 Redis)."""

    async def put(self, key: str, data: bytes) -> TransferResult: ...

    async def get(self, key: str) -> TransferResult: ...


@runtime_checkable
class ModelRepository(Protocol):
    """Fetches model or adapter artifacts to a local path."""

    async def fetch(self, identifier: str, version: str) -> str:
        """Download/locate the artifact and return its local filesystem path."""
        ...


@runtime_checkable
class GPUTransfer(Protocol):
    """Abstracts HBM <-> DRAM data movement (CUDA memcpy)."""

    async def copy_hbm_to_dram(self, hbm_addr: int, cpu_addr: int, size: int) -> TransferResult: ...

    async def copy_dram_to_hbm(self, cpu_addr: int, hbm_addr: int, size: int) -> TransferResult: ...
