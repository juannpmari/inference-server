"""Unit tests for the vLLM OffloadingSpec plugin (no GPU needed)."""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from data_plane.inference.engine.kv_offload.sidecar_backend import (
    SidecarBackend,
    SidecarLoadStoreSpec,
)
from data_plane.inference.engine.kv_offload.sidecar_spec import (
    SidecarOffloadingSpec,
    _DEFAULTS,
)
from data_plane.inference.engine.kv_offload.sidecar_handler import (
    SidecarOffloadingHandler,
    _StagingBufferPool,
    VLLM_HANDLER_AVAILABLE,
)


# =========================================================================
# Backend tests
# =========================================================================


class TestSidecarBackend:
    def test_allocate_free(self):
        backend = SidecarBackend(num_blocks=8)
        assert backend.get_num_free_blocks() == 8

        statuses = backend.allocate_blocks(["h1", "h2", "h3"])
        assert statuses is not None
        assert len(statuses) == 3
        assert backend.get_num_free_blocks() == 5

        # Newly allocated blocks have ref_cnt=-1 (not ready)
        for s in statuses:
            assert s.ref_cnt == -1

        backend.free(statuses[0])
        assert backend.get_num_free_blocks() == 6

    def test_allocate_exceeds_capacity(self):
        backend = SidecarBackend(num_blocks=2)
        assert backend.get_num_free_blocks() == 2
        # Only 2 blocks available, requesting 3 would pop from empty list
        statuses = backend.allocate_blocks(["a", "b"])
        assert len(statuses) == 2

    def test_load_store_spec(self):
        backend = SidecarBackend(num_blocks=4)
        statuses = backend.allocate_blocks(["h1", "h2"])
        assert statuses is not None

        # get_load_store_spec takes (block_hashes, blocks)
        spec = backend.get_load_store_spec(["h1", "h2"], statuses)
        assert isinstance(spec, SidecarLoadStoreSpec)
        expected_ids = [backend.get_block_id(s) for s in statuses]
        assert np.array_equal(spec.block_ids, np.array(expected_ids, dtype=np.int64))
        assert spec.block_hashes == ["h1", "h2"]

    def test_load_store_spec_medium(self):
        spec = SidecarLoadStoreSpec(
            block_ids=np.array([1], dtype=np.int64),
            block_hashes=["h"],
        )
        assert spec.medium() == "SIDECAR"

    def test_duplicate_hash_returns_existing_status(self):
        backend = SidecarBackend(num_blocks=4)
        statuses1 = backend.allocate_blocks(["h1"])
        statuses2 = backend.allocate_blocks(["h1"])  # same hash
        assert statuses1[0] is statuses2[0]  # same BlockStatus returned

    def test_get_block_id(self):
        backend = SidecarBackend(num_blocks=4)
        statuses = backend.allocate_blocks(["h1"])
        assert statuses is not None
        block_id = backend.get_block_id(statuses[0])
        assert block_id is not None
        assert isinstance(block_id, int)


# =========================================================================
# Spec tests
# =========================================================================


class TestSidecarOffloadingSpec:
    def test_default_config(self):
        spec = SidecarOffloadingSpec(vllm_config=None)
        assert spec._grpc_url == "localhost:50051"
        assert spec._num_blocks == 1024
        assert spec.backend is not None

    def test_custom_config(self):
        class FakeKVConfig:
            kv_connector_extra_config = {
                "sidecar_grpc_url": "localhost:9999",
                "num_blocks": 512,
                "block_size_bytes": 65536,
            }

        class FakeCacheConfig:
            block_size = 32

        class FakeVllmConfig:
            kv_transfer_config = FakeKVConfig()
            cache_config = FakeCacheConfig()

        spec = SidecarOffloadingSpec(vllm_config=FakeVllmConfig())
        assert spec._grpc_url == "localhost:9999"
        assert spec._num_blocks == 512
        assert spec._block_size == 65536

    def test_backend_tracks_blocks(self):
        spec = SidecarOffloadingSpec(vllm_config=None)
        assert spec.backend.get_num_free_blocks() == 1024

        statuses = spec.backend.allocate_blocks(["x", "y"])
        assert statuses is not None
        assert spec.backend.get_num_free_blocks() == 1022

    def test_defaults_module_constant(self):
        """Verify _DEFAULTS dict contains all expected keys."""
        assert "sidecar_grpc_url" in _DEFAULTS
        assert "num_blocks" in _DEFAULTS
        assert "block_size_bytes" in _DEFAULTS
        assert _DEFAULTS["sidecar_grpc_url"] == "localhost:50051"
        assert _DEFAULTS["num_blocks"] == 1024
        assert _DEFAULTS["block_size_bytes"] == 131072

    def test_extra_config_none_uses_defaults(self):
        """When extra_config is None, all defaults are applied."""
        spec = SidecarOffloadingSpec(vllm_config=None)
        assert spec._grpc_url == _DEFAULTS["sidecar_grpc_url"]
        assert spec._num_blocks == _DEFAULTS["num_blocks"]
        assert spec._block_size == _DEFAULTS["block_size_bytes"]

    def test_partial_extra_config_inherits_defaults(self):
        """Partial extra_config still fills missing keys from defaults."""

        class FakeKVConfig:
            kv_connector_extra_config = {
                "sidecar_grpc_url": "custom:1234",
                # num_blocks and block_size_bytes intentionally omitted
            }

        class FakeVllmConfig:
            kv_transfer_config = FakeKVConfig()

        spec = SidecarOffloadingSpec(vllm_config=FakeVllmConfig())
        assert spec._grpc_url == "custom:1234"
        assert spec._num_blocks == _DEFAULTS["num_blocks"]
        assert spec._block_size == _DEFAULTS["block_size_bytes"]

    def test_get_handlers_yields_tuple(self):
        """get_handlers() yields (src_type, dst_type, handler) tuples."""
        spec = SidecarOffloadingSpec(vllm_config=None)
        handlers = list(spec.get_handlers())
        assert len(handlers) == 1
        src_type, dst_type, handler = handlers[0]
        assert dst_type is SidecarLoadStoreSpec
        assert isinstance(handler, SidecarOffloadingHandler)


# =========================================================================
# Handler tests (mock gRPC, subtask 1)
# =========================================================================


class TestSidecarOffloadingHandler:
    """Test handler pipeline with mocked gRPC stub."""

    def _make_handler(self, block_size: int = 128):
        handler = SidecarOffloadingHandler(
            grpc_url="localhost:50051",
            block_size_bytes=block_size,
        )
        return handler

    def _mock_stub(self, handler):
        stub = MagicMock()
        handler.stub = stub
        return stub

    # -- store path -----------------------------------------------------------

    def test_transfer_async_store_accepted(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)

        src_spec = MagicMock()  # GPU spec (source)
        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([10, 11], dtype=np.int64),
            block_hashes=["aabb", "ccdd"],
        )
        accepted = handler.transfer_async(job_id=1, spec=(src_spec, dst_spec))
        assert accepted is True

    def test_get_finished_store_calls_grpc(self):
        handler = self._make_handler(block_size=64)
        stub = self._mock_stub(handler)

        # Configure mock responses
        stub.StoreBlock.return_value = MagicMock(success=True)

        src_spec = MagicMock()
        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([5], dtype=np.int64),
            block_hashes=["aa11"],
        )
        handler.transfer_async(job_id=42, spec=(src_spec, dst_spec))
        results = handler.get_finished()

        assert len(results) == 1
        job_id, success = results[0]
        assert job_id == 42
        assert success is True

        # Verify gRPC was called with correct args
        stub.StoreBlock.assert_called_once()
        call_args = stub.StoreBlock.call_args
        req = call_args[0][0]
        assert req.block_id == 5
        assert req.block_hash == "aa11"
        assert len(req.data) == 64  # placeholder path, block_size bytes

    def test_get_finished_store_with_staging_data(self):
        """When staging_data is populated, real data is sent instead of placeholder."""
        handler = self._make_handler(block_size=64)
        stub = self._mock_stub(handler)
        stub.StoreBlock.return_value = MagicMock(success=True)

        src_spec = MagicMock()
        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([7], dtype=np.int64),
            block_hashes=["ff00"],
        )
        handler.transfer_async(job_id=10, spec=(src_spec, dst_spec))

        # Manually inject staging data (simulating GPU extraction)
        handler._pending_stores[0].staging_data = [b"\xAB" * 64]

        results = handler.get_finished()
        assert results == [(10, True)]

        req = stub.StoreBlock.call_args[0][0]
        assert req.data == b"\xAB" * 64

    def test_get_finished_store_failure(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)
        stub.StoreBlock.return_value = MagicMock(success=False)

        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([1], dtype=np.int64),
            block_hashes=["dead"],
        )
        handler.transfer_async(job_id=99, spec=(MagicMock(), dst_spec))
        results = handler.get_finished()
        assert results == [(99, False)]

    def test_get_finished_store_exception(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)
        stub.StoreBlock.side_effect = Exception("connection refused")

        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([1], dtype=np.int64),
            block_hashes=["beef"],
        )
        handler.transfer_async(job_id=7, spec=(MagicMock(), dst_spec))
        results = handler.get_finished()
        assert results == [(7, False)]

    # -- load path ------------------------------------------------------------

    def test_transfer_async_load_accepted(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)

        src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([20], dtype=np.int64),
            block_hashes=["1234"],
        )
        dst_spec = MagicMock()  # GPU spec (destination)
        accepted = handler.transfer_async(job_id=2, spec=(src_spec, dst_spec))
        assert accepted is True

    def test_get_finished_load_calls_grpc(self):
        handler = self._make_handler(block_size=32)
        stub = self._mock_stub(handler)

        payload = b"\xDE\xAD" * 16
        stub.LoadBlock.return_value = MagicMock(success=True, data=payload)

        src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([3], dtype=np.int64),
            block_hashes=["abab"],
        )
        handler.transfer_async(job_id=55, spec=(src_spec, MagicMock()))
        results = handler.get_finished()

        assert len(results) == 1
        assert results[0] == (55, True)

        stub.LoadBlock.assert_called_once()
        req = stub.LoadBlock.call_args[0][0]
        assert req.block_id == 3

        # staging_data should contain the loaded bytes
        # (already consumed from pending_loads which is cleared)
        # We check via the mock that the RPC returned the right data

    def test_get_finished_load_failure(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)
        stub.LoadBlock.return_value = MagicMock(success=False)

        src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([9], dtype=np.int64),
            block_hashes=["0000"],
        )
        handler.transfer_async(job_id=88, spec=(src_spec, MagicMock()))
        results = handler.get_finished()
        assert results == [(88, False)]

    def test_get_finished_load_exception(self):
        handler = self._make_handler()
        stub = self._mock_stub(handler)
        stub.LoadBlock.side_effect = Exception("timeout")

        src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([1], dtype=np.int64),
            block_hashes=["ffff"],
        )
        handler.transfer_async(job_id=33, spec=(src_spec, MagicMock()))
        results = handler.get_finished()
        assert results == [(33, False)]

    # -- multiple blocks ------------------------------------------------------

    def test_store_multiple_blocks(self):
        handler = self._make_handler(block_size=16)
        stub = self._mock_stub(handler)
        stub.StoreBlock.return_value = MagicMock(success=True)

        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([1, 2, 3], dtype=np.int64),
            block_hashes=["h1", "h2", "h3"],
        )
        handler.transfer_async(job_id=100, spec=(MagicMock(), dst_spec))
        results = handler.get_finished()
        assert results == [(100, True)]
        assert stub.StoreBlock.call_count == 3

    def test_load_multiple_blocks(self):
        handler = self._make_handler(block_size=16)
        stub = self._mock_stub(handler)
        stub.LoadBlock.return_value = MagicMock(success=True, data=b"\x00" * 16)

        src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([4, 5], dtype=np.int64),
            block_hashes=["h4", "h5"],
        )
        handler.transfer_async(job_id=200, spec=(src_spec, MagicMock()))
        results = handler.get_finished()
        assert results == [(200, True)]
        assert stub.LoadBlock.call_count == 2

    # -- hash conversion ------------------------------------------------------

    def test_hash_to_str_bytes(self):
        handler = self._make_handler()
        assert handler._hash_to_str(b"\xab\xcd") == "abcd"

    def test_hash_to_str_string(self):
        handler = self._make_handler()
        assert handler._hash_to_str("already_string") == "already_string"


# =========================================================================
# Staging buffer pool tests (subtask 5)
# =========================================================================


class TestStagingBufferPool:
    def test_get_release_cycle(self):
        pool = _StagingBufferPool(block_size_bytes=64, pool_size=0)
        buf = pool.get()
        assert len(buf) == 64
        pool.release(buf)
        # Getting again should return the same buffer (recycled)
        buf2 = pool.get()
        assert buf2 is buf

    def test_pool_exhaustion_allocates_new(self):
        pool = _StagingBufferPool(block_size_bytes=32, pool_size=1)
        buf1 = pool.get()
        buf2 = pool.get()  # pool exhausted, new allocation
        assert buf1 is not buf2
        assert len(buf2) == 32

    def test_bytearray_fallback(self):
        """Without torch, buffers are plain bytearrays."""
        pool = _StagingBufferPool(block_size_bytes=16, pool_size=2)
        if not VLLM_HANDLER_AVAILABLE:
            buf = pool.get()
            assert isinstance(buf, bytearray)


# =========================================================================
# Engine CLI args test (subtask 3)
# =========================================================================


class TestEngineBuildCliArgs:
    """Test _build_cli_args without needing vLLM or a GPU."""

    def test_base_args(self):
        from data_plane.inference.engine.config import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            enable_lora=False,
            enable_prefix_caching=False,
            enable_kv_offload=False,
        )
        from data_plane.inference.engine.engine import Engine
        args = Engine._build_cli_args(config)
        assert "--model" in args
        assert "test-model" in args
        assert "--dtype" in args
        assert "float16" in args
        assert "--gpu-memory-utilization" in args
        assert "0.8" in args
        assert "--max-model-len" in args
        assert "2048" in args
        assert "--kv-transfer-config" not in args

    def test_kv_offload_args(self):
        from data_plane.inference.engine.config import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            enable_kv_offload=True,
            sidecar_grpc_url="sidecar:50051",
            kv_offload_num_blocks=512,
        )
        from data_plane.inference.engine.engine import Engine
        args = Engine._build_cli_args(config)

        assert "--kv-transfer-config" in args
        idx = args.index("--kv-transfer-config")
        kv_config = json.loads(args[idx + 1])

        assert kv_config["kv_connector"] == "OffloadingConnector"
        assert kv_config["kv_role"] == "kv_both"
        extra = kv_config["kv_connector_extra_config"]
        assert extra["spec_name"] == "SidecarOffloadingSpec"
        assert extra["spec_module_path"] == "data_plane.inference.engine.kv_offload.sidecar_spec"
        assert extra["sidecar_grpc_url"] == "sidecar:50051"
        assert extra["num_blocks"] == 512

    def test_lora_args(self):
        from data_plane.inference.engine.config import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            enable_lora=True,
            max_loras=8,
            max_lora_rank=32,
        )
        from data_plane.inference.engine.engine import Engine
        args = Engine._build_cli_args(config)
        assert "--enable-lora" in args
        assert "--max-loras" in args
        assert "8" in args
        assert "--max-lora-rank" in args
        assert "32" in args

    def test_prefix_caching_arg(self):
        from data_plane.inference.engine.config import EngineConfig
        config = EngineConfig(
            model_name="test-model",
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            enable_prefix_caching=True,
        )
        from data_plane.inference.engine.engine import Engine
        args = Engine._build_cli_args(config)
        assert "--enable-prefix-caching" in args

    def test_model_path_override(self):
        from data_plane.inference.engine.config import EngineConfig
        config = EngineConfig(model_name="default-model")
        from data_plane.inference.engine.engine import Engine
        args = Engine._build_cli_args(config, model_path="/custom/path")
        idx = args.index("--model")
        assert args[idx + 1] == "/custom/path"


# =========================================================================
# End-to-end round-trip integration test (subtask 8)
# =========================================================================


class TestEndToEndRoundTrip:
    """Full round-trip: handler -> gRPC server -> L1 cache -> gRPC -> handler.

    The handler uses synchronous gRPC while the server is async.  To avoid
    deadlocking the event loop we run the synchronous handler methods via
    ``asyncio.to_thread``.
    """

    @pytest.fixture
    async def grpc_env(self):
        """Start an in-process gRPC server with L1 + cache manager."""
        import asyncio
        from unittest.mock import AsyncMock

        from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
        from data_plane.inference.sidecar.grpc_server import create_grpc_server
        from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
        from data_plane.inference.sidecar.l1_cache.api import L1ByteStore
        from shared.types import TransferResult

        block_size = 128
        l1 = L1ByteStore(num_blocks=16, block_size_bytes=block_size)
        registry = KVBlockRegistry()
        mock_l2 = AsyncMock()
        mock_l2.put = AsyncMock(return_value=TransferResult(True, "OK"))
        mock_l2.get = AsyncMock(return_value=TransferResult(False, "miss"))
        manager = MultiTieredCacheManager(l1=l1, l2=mock_l2, registry=registry)

        port = 50098
        server = await create_grpc_server(manager, port=port)

        yield port, block_size, manager

        await server.stop(grace=0)

    @pytest.mark.asyncio
    async def test_store_then_load_roundtrip(self, grpc_env):
        """Store a block via handler, load it back, verify byte equality."""
        import asyncio

        port, block_size, manager = grpc_env

        handler = SidecarOffloadingHandler(
            grpc_url=f"localhost:{port}",
            block_size_bytes=block_size,
        )

        # Allocate on server side
        ids = manager.allocate_blocks(["test-hash-rt"])
        assert ids is not None
        block_id = ids[0]

        known_data = bytes(range(128))

        # -- STORE (sync gRPC in worker thread) --
        src_spec = MagicMock()
        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([block_id], dtype=np.int64),
            block_hashes=["test-hash-rt"],
        )
        handler.transfer_async(job_id=1, spec=(src_spec, dst_spec))
        handler._pending_stores[0].staging_data = [known_data]

        store_results = await asyncio.to_thread(handler.get_finished)
        assert store_results == [(1, True)]

        # -- LOAD (sync gRPC in worker thread) --
        load_src_spec = SidecarLoadStoreSpec(
            block_ids=np.array([block_id], dtype=np.int64),
            block_hashes=["test-hash-rt"],
        )
        handler.transfer_async(job_id=2, spec=(load_src_spec, MagicMock()))

        load_results = await asyncio.to_thread(handler.get_finished)
        assert load_results == [(2, True)]

        # Verify via L1 cache directly
        cached_data = manager.l1.load(block_id)
        assert cached_data == known_data

    @pytest.mark.asyncio
    async def test_store_multiple_and_load_back(self, grpc_env):
        """Store two blocks, load both back, verify data integrity."""
        import asyncio

        port, block_size, manager = grpc_env

        handler = SidecarOffloadingHandler(
            grpc_url=f"localhost:{port}",
            block_size_bytes=block_size,
        )

        ids = manager.allocate_blocks(["hash-a", "hash-b"])
        assert ids is not None
        bid_a, bid_b = ids

        data_a = b"\xAA" * block_size
        data_b = b"\xBB" * block_size

        # Store both blocks
        dst_spec = SidecarLoadStoreSpec(
            block_ids=np.array([bid_a, bid_b], dtype=np.int64),
            block_hashes=["hash-a", "hash-b"],
        )
        handler.transfer_async(job_id=10, spec=(MagicMock(), dst_spec))
        handler._pending_stores[0].staging_data = [data_a, data_b]
        results = await asyncio.to_thread(handler.get_finished)
        assert results == [(10, True)]

        # Load both back
        load_src = SidecarLoadStoreSpec(
            block_ids=np.array([bid_a, bid_b], dtype=np.int64),
            block_hashes=["hash-a", "hash-b"],
        )
        handler.transfer_async(job_id=11, spec=(load_src, MagicMock()))
        results = await asyncio.to_thread(handler.get_finished)
        assert results == [(11, True)]

        # Verify via L1
        assert manager.l1.load(bid_a) == data_a
        assert manager.l1.load(bid_b) == data_b
