"""gRPC server module for the sidecar's KV cache service."""

import logging

import grpc

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.kv_cache_api import KVCacheServicer
from shared.proto import kv_cache_pb2_grpc

logger = logging.getLogger(__name__)

# 16 MB max message size (KV blocks can be large)
_MAX_MESSAGE_SIZE = 16 * 1024 * 1024

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
]


async def create_grpc_server(
    cache_manager: MultiTieredCacheManager,
    port: int = 50051,
) -> grpc.aio.Server:
    """Create, configure, and start the gRPC server."""
    server = grpc.aio.server(options=_GRPC_OPTIONS)
    servicer = KVCacheServicer(cache_manager)
    kv_cache_pb2_grpc.add_KVCacheServiceServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info(f"gRPC KV cache server started on {listen_addr}")
    return server
