"""vLLM OffloadingSpec plugin that configures sidecar-based KV cache offloading.

Registered via --kv-transfer-config in the engine CLI args. vLLM's scheduler
calls get_manager() for offload decisions and get_handlers() for the actual
data movement.
"""

import logging
from typing import Generator, Optional

from data_plane.inference.engine.kv_offload.sidecar_backend import (
    SidecarBackend,
    SidecarLoadStoreSpec,
)
from data_plane.inference.engine.kv_offload.sidecar_handler import SidecarOffloadingHandler

logger = logging.getLogger(__name__)

try:
    from vllm.v1.kv_offload.spec import OffloadingSpec
    from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        GPULoadStoreSpec,
    )
    VLLM_SPEC_AVAILABLE = True
except ImportError:
    VLLM_SPEC_AVAILABLE = False

    class OffloadingSpec:
        def __init__(self, vllm_config=None):
            self.extra_config = {}
            self.gpu_block_size = 16
            self.offloaded_block_size = 16

    class GPULoadStoreSpec:
        pass


class SidecarOffloadingSpec(OffloadingSpec):
    """Configures vLLM to offload KV blocks to the sidecar over gRPC.

    Reads configuration from extra_config (set by parent from
    vllm_config.kv_transfer_config.kv_connector_extra_config):
        - sidecar_grpc_url: gRPC address of the sidecar (default: localhost:50051)
        - num_blocks: number of block slots in the sidecar (default: 1024)
        - block_size_bytes: size of each block (default: 131072 = 128KB)
    """

    def __init__(self, vllm_config=None):
        super().__init__(vllm_config)

        extra = getattr(self, "extra_config", None) or {}
        self._grpc_url = extra.get("sidecar_grpc_url", "localhost:50051")
        self._num_blocks = int(extra.get("num_blocks", 1024))
        self._block_size = int(extra.get("block_size_bytes", 131072))

        block_size = getattr(self, "offloaded_block_size", 16)
        self._backend = SidecarBackend(self._num_blocks, block_size=block_size)

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
        """Yield (src_type, dst_type, handler) tuples for the scheduler."""
        handler = SidecarOffloadingHandler(
            grpc_url=self._grpc_url,
            block_size_bytes=self._block_size,
        )
        yield (GPULoadStoreSpec, SidecarLoadStoreSpec, handler)

    @property
    def backend(self) -> SidecarBackend:
        return self._backend
