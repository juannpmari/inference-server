"""vLLM OffloadingSpec plugin that configures sidecar-based KV cache offloading.

Registered via --kv-transfer-config in the engine CLI args. vLLM's scheduler
calls get_manager() for offload decisions and get_handlers() for the actual
data movement.
"""

import logging
from typing import Generator, Optional

from data_plane.inference.engine.kv_offload.sidecar_backend import SidecarBackend
from data_plane.inference.engine.kv_offload.sidecar_handler import SidecarOffloadingHandler
from data_plane.inference.engine.sidecar_cache_client import SidecarCacheClient

logger = logging.getLogger(__name__)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        OffloadingSpec,
        OffloadingHandler,
    )
    from vllm.v1.core.kv_cache_manager import LRUOffloadingManager
    VLLM_SPEC_AVAILABLE = True
except ImportError:
    VLLM_SPEC_AVAILABLE = False

    class OffloadingSpec:
        pass


class SidecarOffloadingSpec(OffloadingSpec):
    """Configures vLLM to offload KV blocks to the sidecar over gRPC.

    Reads configuration from VllmConfig.kv_transfer_config.kv_connector_extra_config:
        - sidecar_grpc_url: gRPC address of the sidecar (default: localhost:50051)
        - num_blocks: number of block slots in the sidecar (default: 1024)
        - block_size_bytes: size of each block (default: 131072 = 128KB)
    """

    def __init__(self, vllm_config=None):
        self._vllm_config = vllm_config
        extra = {}
        if vllm_config is not None:
            kv_config = getattr(vllm_config, "kv_transfer_config", None)
            if kv_config is not None:
                extra = getattr(kv_config, "kv_connector_extra_config", {}) or {}

        self._grpc_url = extra.get("sidecar_grpc_url", "localhost:50051")
        self._num_blocks = int(extra.get("num_blocks", 1024))
        self._block_size = int(extra.get("block_size_bytes", 131072))

        self._backend = SidecarBackend(self._num_blocks)
        self._client = SidecarCacheClient(grpc_url=self._grpc_url)

        logger.info(
            f"SidecarOffloadingSpec: url={self._grpc_url}, "
            f"blocks={self._num_blocks}, block_size={self._block_size}"
        )

    def get_manager(self):
        """Return an LRUOffloadingManager backed by SidecarBackend."""
        if not VLLM_SPEC_AVAILABLE:
            raise RuntimeError("vLLM offloading APIs not available")
        return LRUOffloadingManager(self._backend)

    def get_handlers(self, kv_caches=None) -> Generator:
        """Yield (gpu_spec, sidecar_spec, handler) tuples for the scheduler."""
        handler = SidecarOffloadingHandler(
            client=self._client,
            block_size_bytes=self._block_size,
        )
        yield handler

    @property
    def backend(self) -> SidecarBackend:
        return self._backend

    @property
    def client(self) -> SidecarCacheClient:
        return self._client
