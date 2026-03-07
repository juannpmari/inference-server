"""gRPC client for the sidecar's KV cache service.

Used by the engine (both MockEngine and vLLM OffloadingHandler) to store/load
KV block bytes over gRPC to the sidecar container.
"""

import logging
from typing import Optional

import grpc

from shared.proto import kv_cache_pb2, kv_cache_pb2_grpc

logger = logging.getLogger(__name__)

# 16 MB max message size
_MAX_MESSAGE_SIZE = 16 * 1024 * 1024

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
]


class SidecarCacheClient:
    """Async gRPC client wrapping KVCacheService RPCs."""

    def __init__(self, grpc_url: str = "localhost:50051"):
        self._grpc_url = grpc_url
        self._channel = grpc.aio.insecure_channel(grpc_url, options=_GRPC_OPTIONS)
        self._stub = kv_cache_pb2_grpc.KVCacheServiceStub(self._channel)
        logger.info(f"SidecarCacheClient initialized (grpc_url={grpc_url})")

    async def store_block(
        self,
        block_id: int,
        block_hash: str,
        data: bytes,
        model_id: str = "",
        layer_name: str = "",
    ) -> bool:
        """Send block bytes to sidecar for storage."""
        try:
            resp = await self._stub.StoreBlock(
                kv_cache_pb2.StoreBlockRequest(
                    block_id=block_id,
                    block_hash=block_hash,
                    data=data,
                    model_id=model_id,
                    layer_name=layer_name,
                ),
                timeout=5.0,
            )
            return resp.success
        except grpc.RpcError as e:
            logger.error(f"StoreBlock RPC failed: {e}")
            return False

    async def load_block(self, block_id: int, layer_name: str = "") -> Optional[bytes]:
        """Request block bytes from sidecar."""
        try:
            resp = await self._stub.LoadBlock(
                kv_cache_pb2.LoadBlockRequest(
                    block_id=block_id,
                    layer_name=layer_name,
                ),
                timeout=5.0,
            )
            if resp.success:
                return resp.data
            return None
        except grpc.RpcError as e:
            logger.error(f"LoadBlock RPC failed: {e}")
            return None

    async def get_free_blocks(self) -> int:
        """Query available capacity."""
        try:
            resp = await self._stub.GetFreeBlocks(
                kv_cache_pb2.GetFreeBlocksRequest(),
                timeout=5.0,
            )
            return resp.num_free_blocks
        except grpc.RpcError as e:
            logger.error(f"GetFreeBlocks RPC failed: {e}")
            return 0

    async def allocate_blocks(self, block_hashes: list[str]) -> Optional[list[int]]:
        """Reserve block slots by hash."""
        try:
            resp = await self._stub.AllocateBlocks(
                kv_cache_pb2.AllocateBlocksRequest(block_hashes=block_hashes),
                timeout=5.0,
            )
            if resp.success:
                return list(resp.block_ids)
            return None
        except grpc.RpcError as e:
            logger.error(f"AllocateBlocks RPC failed: {e}")
            return None

    async def free_block(self, block_id: int) -> bool:
        """Release a block slot."""
        try:
            resp = await self._stub.FreeBlock(
                kv_cache_pb2.FreeBlockRequest(block_id=block_id),
                timeout=5.0,
            )
            return resp.success
        except grpc.RpcError as e:
            logger.error(f"FreeBlock RPC failed: {e}")
            return False

    async def close(self):
        """Close the gRPC channel."""
        await self._channel.close()
