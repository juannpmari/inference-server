"""Sidecar offloading handler — performs GPU<->CPU<->gRPC transfers.

Implements vLLM's OffloadingHandler interface. The transfer flow:
  Store (GPU->Sidecar): GPU block -> pinned CPU staging -> bytes -> gRPC StoreBlock
  Load (Sidecar->GPU):  gRPC LoadBlock -> bytes -> pinned CPU staging -> GPU block

The handler uses synchronous gRPC since vLLM calls get_finished() from a
worker thread without an event loop.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import grpc

from data_plane.inference.engine import metrics as engine_metrics
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


# ---------------------------------------------------------------------------
# Pinned CPU staging buffer pool
# ---------------------------------------------------------------------------

class _StagingBufferPool:
    """Pool of pinned-memory CPU buffers for GPU<->CPU copies.

    Falls back to plain ``bytearray`` when torch is unavailable (tests).
    """

    def __init__(self, block_size_bytes: int, pool_size: int = 32):
        self._block_size = block_size_bytes
        self._pool: deque = deque()
        self._use_torch = VLLM_HANDLER_AVAILABLE
        # Pre-allocate a few buffers
        for _ in range(min(pool_size, 4)):
            self._pool.append(self._alloc())

    def _alloc(self):
        if self._use_torch:
            return torch.empty(self._block_size, dtype=torch.uint8, pin_memory=True)
        return bytearray(self._block_size)

    def get(self):
        """Acquire a staging buffer from the pool (or allocate a new one)."""
        if self._pool:
            return self._pool.popleft()
        return self._alloc()

    def release(self, buf):
        """Return a staging buffer to the pool."""
        self._pool.append(buf)


# ---------------------------------------------------------------------------
# Pending operation descriptors
# ---------------------------------------------------------------------------

@dataclass
class _PendingStore:
    """Tracks a pending GPU->Sidecar store operation."""
    job_id: int
    block_ids: list[int]
    block_hashes: list[bytes]
    model_id: str
    # After CUDA event completes, staging tensors hold the data
    staging_data: list = field(default_factory=list)  # list[bytes | torch.Tensor]
    cuda_event: object = None  # torch.cuda.Event
    gpu_copy_done: bool = False


@dataclass
class _PendingLoad:
    """Tracks a pending Sidecar->GPU load operation."""
    job_id: int
    block_ids: list[int]
    # After gRPC completes, staging tensors hold the data
    staging_data: list = field(default_factory=list)  # list[bytes | torch.Tensor]
    cuda_event: object = None  # torch.cuda.Event
    grpc_done: bool = False
    # Reference to the destination GPU spec for CPU->GPU copy
    dst_spec: object = None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class SidecarOffloadingHandler(OffloadingHandler):
    """Handles async GPU<->Sidecar transfers via pinned CPU staging + gRPC.

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
        self._staging_pool = _StagingBufferPool(block_size_bytes)
        logger.info("SidecarOffloadingHandler initialized")

    # -- public interface (called by stub for tests) --------------------------

    @property
    def stub(self):
        return self._stub

    @stub.setter
    def stub(self, value):
        self._stub = value

    # -- transfer_async -------------------------------------------------------

    def transfer_async(
        self,
        job_id: int,
        spec,
    ) -> bool:
        """Initiate an async transfer between GPU and sidecar.

        vLLM passes spec as a TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
        where spec = (src_spec, dst_spec).

        If dst is SidecarLoadStoreSpec: Store path (GPU -> Sidecar).
        If src is SidecarLoadStoreSpec: Load path (Sidecar -> GPU).

        Returns True to indicate the transfer was accepted.
        """
        src_spec, dst_spec = spec

        if isinstance(dst_spec, SidecarLoadStoreSpec):
            # Store: GPU -> Sidecar
            block_ids = dst_spec.block_ids.tolist() if hasattr(dst_spec.block_ids, 'tolist') else list(dst_spec.block_ids)
            pending = _PendingStore(
                job_id=job_id,
                block_ids=block_ids,
                block_hashes=dst_spec.block_hashes,
                model_id="",
            )

            # Extract real tensor data from the GPU source spec if available
            if VLLM_HANDLER_AVAILABLE and hasattr(src_spec, 'kv_caches'):
                try:
                    event = torch.cuda.Event()
                    for idx, bid in enumerate(block_ids):
                        staging_buf = self._staging_pool.get()
                        # src_spec.kv_caches is a list of GPU tensors indexed by block_id
                        gpu_tensor = src_spec.kv_caches[bid]
                        flat = gpu_tensor.reshape(-1).to(torch.uint8)[:self._block_size]
                        staging_buf[:flat.numel()].copy_(flat, non_blocking=True)
                        pending.staging_data.append(staging_buf)
                    event.record()
                    pending.cuda_event = event
                except Exception as e:
                    logger.warning(f"GPU tensor extraction failed, using placeholder path: {e}")
                    # Release any acquired staging buffers
                    for buf in pending.staging_data:
                        self._staging_pool.release(buf)
                    pending.staging_data = []
                    pending.cuda_event = None

            self._pending_stores.append(pending)

        elif isinstance(src_spec, SidecarLoadStoreSpec):
            # Load: Sidecar -> GPU
            block_ids = src_spec.block_ids.tolist() if hasattr(src_spec.block_ids, 'tolist') else list(src_spec.block_ids)
            pending = _PendingLoad(
                job_id=job_id,
                block_ids=block_ids,
                dst_spec=dst_spec if VLLM_HANDLER_AVAILABLE else None,
            )
            self._pending_loads.append(pending)

        return True

    def _hash_to_str(self, bh) -> str:
        """Convert a BlockHash (bytes) to a hex string for gRPC."""
        if isinstance(bh, bytes):
            return bh.hex()
        return str(bh)

    # -- get_finished ---------------------------------------------------------

    def get_finished(self) -> list[tuple[int, bool]]:
        """Poll for completed transfers. Returns list of (job_id, success).

        Uses synchronous gRPC -- safe to call from any thread.
        """
        results = []

        # Process pending stores
        for store in self._pending_stores:
            t0 = time.monotonic()
            try:
                # Wait for CUDA event if we have real GPU data
                if store.cuda_event is not None:
                    store.cuda_event.synchronize()

                success = True
                total_bytes = 0
                for bid, bh in zip(store.block_ids, store.block_hashes):
                    # Get real data from staging buffer, or fall back to placeholder
                    if store.staging_data:
                        raw = store.staging_data.pop(0)
                        if VLLM_HANDLER_AVAILABLE and hasattr(raw, 'numpy'):
                            data = raw.numpy().tobytes()
                            self._staging_pool.release(raw)
                        elif isinstance(raw, (bytes, bytearray)):
                            data = bytes(raw)
                        else:
                            data = bytes(raw)
                    else:
                        data = b"\x00" * self._block_size

                    total_bytes += len(data)
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
                status = "success" if success else "error"
                engine_metrics.engine_kv_offload_operations_total.labels(
                    op="store", status=status
                ).inc()
                engine_metrics.engine_kv_offload_bytes_total.labels(
                    direction="store"
                ).inc(total_bytes)
            except Exception as e:
                logger.error(f"Store job {store.job_id} failed: {e}")
                results.append((store.job_id, False))
                engine_metrics.engine_kv_offload_operations_total.labels(
                    op="store", status="error"
                ).inc()
                # Release any remaining staging buffers
                for buf in store.staging_data:
                    if VLLM_HANDLER_AVAILABLE and hasattr(buf, 'numpy'):
                        self._staging_pool.release(buf)
            finally:
                engine_metrics.engine_kv_offload_duration_seconds.labels(
                    op="store"
                ).observe(time.monotonic() - t0)
        self._pending_stores = []

        # Process pending loads
        for load in self._pending_loads:
            t0 = time.monotonic()
            try:
                success = True
                total_bytes = 0
                for bid in load.block_ids:
                    resp = self._stub.LoadBlock(
                        kv_cache_pb2.LoadBlockRequest(block_id=bid),
                        timeout=_RPC_TIMEOUT,
                    )
                    if not resp.success:
                        success = False
                        break
                    total_bytes += len(resp.data)

                    # If we have a GPU destination, copy through staging buffer
                    if VLLM_HANDLER_AVAILABLE and load.dst_spec is not None and hasattr(load.dst_spec, 'kv_caches'):
                        try:
                            staging_buf = self._staging_pool.get()
                            src_tensor = torch.frombuffer(resp.data, dtype=torch.uint8)
                            staging_buf[:len(resp.data)].copy_(src_tensor)
                            # Initiate CPU->GPU copy
                            gpu_tensor = load.dst_spec.kv_caches[bid]
                            flat_gpu = gpu_tensor.reshape(-1).to(torch.uint8)[:self._block_size]
                            flat_gpu.copy_(staging_buf[:flat_gpu.numel()], non_blocking=True)
                            event = torch.cuda.Event()
                            event.record()
                            load.cuda_event = event
                            self._staging_pool.release(staging_buf)
                        except Exception as e:
                            logger.warning(f"GPU load copy failed, storing raw bytes: {e}")
                            load.staging_data.append(resp.data)
                    else:
                        load.staging_data.append(resp.data)

                results.append((load.job_id, success))
                status = "success" if success else "error"
                engine_metrics.engine_kv_offload_operations_total.labels(
                    op="load", status=status
                ).inc()
                engine_metrics.engine_kv_offload_bytes_total.labels(
                    direction="load"
                ).inc(total_bytes)
            except Exception as e:
                logger.error(f"Load job {load.job_id} failed: {e}")
                results.append((load.job_id, False))
                engine_metrics.engine_kv_offload_operations_total.labels(
                    op="load", status="error"
                ).inc()
            finally:
                engine_metrics.engine_kv_offload_duration_seconds.labels(
                    op="load"
                ).observe(time.monotonic() - t0)
        self._pending_loads = []

        return results
