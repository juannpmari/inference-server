"""gRPC servicer for KV Cache operations.

Implements kv_cache.KVCacheService as a real gRPC servicer.
Each RPC method delegates to MultiTieredCacheManager.
"""

import logging

import grpc

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from shared.proto import kv_cache_pb2, kv_cache_pb2_grpc

logger = logging.getLogger(__name__)


class KVCacheServicer(kv_cache_pb2_grpc.KVCacheServiceServicer):
    """gRPC servicer exposing cache store/load/allocate/free operations."""

    def __init__(self, cache_manager: MultiTieredCacheManager):
        self._manager = cache_manager
        logger.info("KVCacheServicer initialized")

    async def StoreBlock(self, request, context):
        try:
            ok = await self._manager.store_block(
                block_id=request.block_id,
                block_hash=request.block_hash,
                data=request.data,
                model_id=request.model_id,
                layer_name=request.layer_name,
            )
            return kv_cache_pb2.StoreBlockResponse(
                success=ok,
                message="stored" if ok else "store failed",
            )
        except Exception as e:
            logger.error(f"StoreBlock error: {e}")
            return kv_cache_pb2.StoreBlockResponse(success=False, message=str(e))

    async def LoadBlock(self, request, context):
        try:
            data = await self._manager.load_block(request.block_id)
            if data is not None:
                return kv_cache_pb2.LoadBlockResponse(
                    success=True, data=data, message="loaded"
                )
            return kv_cache_pb2.LoadBlockResponse(
                success=False, data=b"", message="block not found"
            )
        except Exception as e:
            logger.error(f"LoadBlock error: {e}")
            return kv_cache_pb2.LoadBlockResponse(
                success=False, data=b"", message=str(e)
            )

    async def GetFreeBlocks(self, request, context):
        num_free = self._manager.get_num_free_blocks()
        return kv_cache_pb2.GetFreeBlocksResponse(num_free_blocks=num_free)

    async def AllocateBlocks(self, request, context):
        try:
            ids = self._manager.allocate_blocks(list(request.block_hashes))
            if ids is not None:
                return kv_cache_pb2.AllocateBlocksResponse(
                    success=True, block_ids=ids, message="allocated"
                )
            return kv_cache_pb2.AllocateBlocksResponse(
                success=False, block_ids=[], message="insufficient capacity"
            )
        except Exception as e:
            logger.error(f"AllocateBlocks error: {e}")
            return kv_cache_pb2.AllocateBlocksResponse(
                success=False, block_ids=[], message=str(e)
            )

    async def FreeBlock(self, request, context):
        try:
            ok = self._manager.free_block(request.block_id)
            return kv_cache_pb2.FreeBlockResponse(
                success=ok,
                message="freed" if ok else "block not found",
            )
        except Exception as e:
            logger.error(f"FreeBlock error: {e}")
            return kv_cache_pb2.FreeBlockResponse(success=False, message=str(e))
