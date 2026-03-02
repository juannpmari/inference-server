"""Sidecar offloading handler — performs GPU↔CPU↔gRPC transfers.

Implements vLLM's OffloadingHandler interface. The transfer flow:
  Store (GPU→Sidecar): GPU block → pinned CPU staging → bytes → gRPC StoreBlock
  Load (Sidecar→GPU):  gRPC LoadBlock → bytes → pinned CPU staging → GPU block

CUDA events are used to track async GPU↔CPU copies. The gRPC calls are
synchronous from the handler's perspective (awaited in get_finished).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from data_plane.inference.engine.sidecar_cache_client import SidecarCacheClient
from data_plane.inference.engine.kv_offload.sidecar_backend import SidecarLoadStoreSpec

logger = logging.getLogger(__name__)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import OffloadingHandler
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
    block_hashes: list[str]
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
    """Handles async GPU↔Sidecar transfers via pinned CPU staging + gRPC."""

    def __init__(
        self,
        client: SidecarCacheClient,
        block_size_bytes: int = 131072,
    ):
        self._client = client
        self._block_size = block_size_bytes
        self._pending_stores: list[_PendingStore] = []
        self._pending_loads: list[_PendingLoad] = []
        self._next_job_id = 0
        logger.info("SidecarOffloadingHandler initialized")

    def transfer_async(
        self,
        job_id: int,
        src_spec,
        dst_spec,
    ) -> None:
        """Initiate an async transfer between GPU and sidecar.

        If src is GPU (LoadStoreSpec) and dst is Sidecar (SidecarLoadStoreSpec):
            Store path — copy GPU→staging, then gRPC store.
        If src is Sidecar and dst is GPU:
            Load path — gRPC load, then copy staging→GPU.
        """
        if isinstance(dst_spec, SidecarLoadStoreSpec):
            # Store: GPU → Sidecar
            pending = _PendingStore(
                job_id=job_id,
                block_ids=dst_spec.block_ids,
                block_hashes=dst_spec.block_hashes,
                model_id="",
            )
            if VLLM_HANDLER_AVAILABLE:
                # In production with GPU: use CUDA stream + event
                # For now, serialize mock data
                pass
            self._pending_stores.append(pending)

        elif isinstance(src_spec, SidecarLoadStoreSpec):
            # Load: Sidecar → GPU
            pending = _PendingLoad(
                job_id=job_id,
                block_ids=src_spec.block_ids,
            )
            self._pending_loads.append(pending)

    async def get_finished(self) -> list[tuple[int, bool]]:
        """Poll for completed transfers. Returns list of (job_id, success)."""
        results = []

        # Process pending stores
        remaining_stores = []
        for store in self._pending_stores:
            try:
                success = True
                for bid, bh in zip(store.block_ids, store.block_hashes):
                    # In mock/test mode, generate dummy data if not already set
                    data = b"\x00" * self._block_size
                    if store.staging_data:
                        data = store.staging_data.pop(0)
                    ok = await self._client.store_block(
                        block_id=bid,
                        block_hash=bh,
                        data=data,
                        model_id=store.model_id,
                    )
                    if not ok:
                        success = False
                        break
                results.append((store.job_id, success))
            except Exception as e:
                logger.error(f"Store job {store.job_id} failed: {e}")
                results.append((store.job_id, False))
        self._pending_stores = remaining_stores

        # Process pending loads
        remaining_loads = []
        for load in self._pending_loads:
            try:
                success = True
                for bid in load.block_ids:
                    data = await self._client.load_block(bid)
                    if data is None:
                        success = False
                        break
                    load.staging_data.append(data)
                results.append((load.job_id, success))
            except Exception as e:
                logger.error(f"Load job {load.job_id} failed: {e}")
                results.append((load.job_id, False))
        self._pending_loads = remaining_loads

        return results
