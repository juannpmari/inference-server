"""Client for calling the sidecar's KV cache gRPC service from the engine.

In production, vLLM's OffloadingConnector would call these methods. In mock mode,
MockEngine calls them directly to simulate the offload/fetch flow.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from shared.types import BlockReference, TransferResult

logger = logging.getLogger(__name__)


@dataclass
class OffloadRequest:
    """Request payload for offloading a KV block."""
    key: str
    device_id: int
    memory_address: int
    size_bytes: int
    model_id: str = ""
    prefix_hash: str = ""


@dataclass
class FetchRequest:
    """Request payload for fetching a KV block back to HBM."""
    key: str
    dest_device_id: int
    dest_memory_address: int
    dest_size_bytes: int


@dataclass
class SpaceRequest:
    """Request payload for L1 eviction."""
    needed_size_bytes: int


class SidecarCacheClient:
    """Wraps gRPC calls to the sidecar's KVCacheAPIService.

    In the current implementation, this client holds a direct reference to the
    KVCacheAPIService for in-process communication (engine and sidecar co-located
    in the same pod). A future version can replace this with actual gRPC stubs.
    """

    def __init__(self, kv_service=None, grpc_url: str = "localhost:50051"):
        self._service = kv_service
        self._grpc_url = grpc_url
        logger.info(f"SidecarCacheClient initialized (grpc_url={grpc_url})")

    async def offload_block(
        self,
        key: str,
        block_ref: BlockReference,
        model_id: str = "",
        prefix_hash: str = "",
    ) -> TransferResult:
        """Ask sidecar to offload a KV block from HBM."""
        if not self._service:
            return TransferResult(False, "No KV cache service configured")

        request = OffloadRequest(
            key=key,
            device_id=block_ref.device_id,
            memory_address=block_ref.memory_address,
            size_bytes=block_ref.size_bytes,
            model_id=model_id,
            prefix_hash=prefix_hash,
        )
        return await self._service.OffloadKVBlock(request, context=None)

    async def fetch_block(self, key: str, dest_ref: BlockReference) -> TransferResult:
        """Ask sidecar to fetch a KV block back into HBM."""
        if not self._service:
            return TransferResult(False, "No KV cache service configured")

        request = FetchRequest(
            key=key,
            dest_device_id=dest_ref.device_id,
            dest_memory_address=dest_ref.memory_address,
            dest_size_bytes=dest_ref.size_bytes,
        )
        return await self._service.FetchKVBlock(request, context=None)

    async def check_availability(self, key: str) -> bool:
        """Check if a key exists in any cache tier."""
        if not self._service:
            return False

        @dataclass
        class _Req:
            key: str

        return await self._service.QueryAvailability(_Req(key=key), context=None)

    async def request_l1_space(self, needed_bytes: int) -> TransferResult:
        """Force L1 eviction to free space."""
        if not self._service:
            return TransferResult(False, "No KV cache service configured")

        return await self._service.RequestL1Space(
            SpaceRequest(needed_size_bytes=needed_bytes), context=None
        )
