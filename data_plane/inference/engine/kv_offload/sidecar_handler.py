"""Sidecar offloading handler — performs GPU↔CPU↔gRPC transfers.

Implements vLLM's OffloadingHandler interface. The transfer flow:
  Store (GPU→Sidecar): GPU block → pinned CPU staging → bytes → gRPC StoreBlock
  Load (Sidecar→GPU):  gRPC LoadBlock → bytes → pinned CPU staging → GPU block

The handler uses synchronous gRPC since vLLM calls get_finished() from a
worker thread without an event loop.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import grpc

from shared.config_loader import get_config
from shared.proto import kv_cache_pb2, kv_cache_pb2_grpc
from data_plane.inference.engine.kv_offload.sidecar_backend import SidecarLoadStoreSpec

logger = logging.getLogger(__name__)

_grpc_cfg = get_config("grpc")
_MAX_MESSAGE_SIZE = _grpc_cfg.get("max_message_size", 16 * 1024 * 1024)
_RPC_TIMEOUT = _grpc_cfg.get("rpc_timeout", 5.0)
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
]

try:
    from vllm.v1.kv_offload.worker.worker import OffloadingHandler
    import torch
    import numpy as np
    VLLM_HANDLER_AVAILABLE = True
except ImportError:
    VLLM_HANDLER_AVAILABLE = False

    class OffloadingHandler:
        pass


@dataclass
class _PendingStore:
    """Tracks a pending GPU→Sidecar store operation."""
    job_id: int
    block_ids: list[int]
    block_hashes: list[bytes]
    model_id: str
    # After CUDA event completes, staging tensors hold the data
    staging_data: list[bytes] = field(default_factory=list)
    cuda_event: object = None  # torch.cuda.Event
    gpu_copy_done: bool = False


@dataclass
class _PendingLoad:
    """Tracks a pending Sidecar→GPU load operation."""
    job_id: int
    block_ids: list[int]
    # After gRPC completes, staging tensors hold the data
    staging_data: list[bytes] = field(default_factory=list)
    cuda_event: object = None  # torch.cuda.Event
    grpc_done: bool = False


class SidecarOffloadingHandler(OffloadingHandler):
    """Handles async GPU↔Sidecar transfers via pinned CPU staging + gRPC.

    Uses synchronous gRPC because vLLM calls get_finished() from a worker
    thread without an asyncio event loop.
    """

    def __init__(
        self,
        grpc_url: str = "sidecar:50051",
        block_size_bytes: int = 131072,
    ):
        self._grpc_url = grpc_url
        self._block_size = block_size_bytes
        self._pending_stores: list[_PendingStore] = []
        self._pending_loads: list[_PendingLoad] = []
        self._channel = grpc.insecure_channel(grpc_url, options=_GRPC_OPTIONS)
        self._stub = kv_cache_pb2_grpc.KVCacheServiceStub(self._channel)
        logger.info("SidecarOffloadingHandler initialized")

    def transfer_async(
        self,
        job_id: int,
        spec,
    ) -> bool:
        """Initiate an async transfer between GPU and sidecar.

        vLLM passes spec as a TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
        where spec = (src_spec, dst_spec).

        If dst is SidecarLoadStoreSpec: Store path (GPU → Sidecar).
        If src is SidecarLoadStoreSpec: Load path (Sidecar → GPU).

        Returns True to indicate the transfer was accepted.
        """
        src_spec, dst_spec = spec

        if isinstance(dst_spec, SidecarLoadStoreSpec):
            # Store: GPU → Sidecar
            block_ids = dst_spec.block_ids.tolist() if hasattr(dst_spec.block_ids, 'tolist') else list(dst_spec.block_ids)
            pending = _PendingStore(
                job_id=job_id,
                block_ids=block_ids,
                block_hashes=dst_spec.block_hashes,
                model_id="",
            )
            self._pending_stores.append(pending)

        elif isinstance(src_spec, SidecarLoadStoreSpec):
            # Load: Sidecar → GPU
            block_ids = src_spec.block_ids.tolist() if hasattr(src_spec.block_ids, 'tolist') else list(src_spec.block_ids)
            pending = _PendingLoad(
                job_id=job_id,
                block_ids=block_ids,
            )
            self._pending_loads.append(pending)

        return True

    def _hash_to_str(self, bh) -> str:
        """Convert a BlockHash (bytes) to a hex string for gRPC."""
        if isinstance(bh, bytes):
            return bh.hex()
        return str(bh)

    def get_finished(self) -> list[tuple[int, bool]]:
        """Poll for completed transfers. Returns list of (job_id, success).

        Uses synchronous gRPC — safe to call from any thread.
        """
        results = []

        # Process pending stores
        for store in self._pending_stores:
            try:
                success = True
                for bid, bh in zip(store.block_ids, store.block_hashes):
                    data = b"\x00" * self._block_size
                    if store.staging_data:
                        data = store.staging_data.pop(0)
                    resp = self._stub.StoreBlock(
                        kv_cache_pb2.StoreBlockRequest(
                            block_id=bid,
                            block_hash=self._hash_to_str(bh),
                            data=data,
                            model_id=store.model_id,
                        ),
                        timeout=_RPC_TIMEOUT,
                    )
                    if not resp.success:
                        success = False
                        break
                results.append((store.job_id, success))
            except Exception as e:
                logger.error(f"Store job {store.job_id} failed: {e}")
                results.append((store.job_id, False))
        self._pending_stores = []

        # Process pending loads
        for load in self._pending_loads:
            try:
                success = True
                for bid in load.block_ids:
                    resp = self._stub.LoadBlock(
                        kv_cache_pb2.LoadBlockRequest(block_id=bid),
                        timeout=_RPC_TIMEOUT,
                    )
                    if not resp.success:
                        success = False
                        break
                    load.staging_data.append(resp.data)
                results.append((load.job_id, success))
            except Exception as e:
                logger.error(f"Load job {load.job_id} failed: {e}")
                results.append((load.job_id, False))
        self._pending_loads = []

        return results
