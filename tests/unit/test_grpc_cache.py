"""gRPC integration tests — in-process server + client for KV cache service."""

from unittest.mock import AsyncMock

import grpc
import pytest

from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.grpc_server import create_grpc_server
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from data_plane.inference.sidecar.l1_cache.api import L1ByteStore
from data_plane.inference.engine.sidecar_cache_client import SidecarCacheClient
from data_plane.inference.engine.mock_engine import MockLLMEngine
from shared.types import TransferResult


@pytest.fixture
def mock_l2():
    l2 = AsyncMock()
    l2.put = AsyncMock(return_value=TransferResult(True, "L2 OK"))
    l2.get = AsyncMock(return_value=TransferResult(False, "L2 miss"))
    return l2


@pytest.fixture
async def grpc_env(mock_l2):
    """Start an in-process gRPC server, yield a connected client, then shut down."""
    l1 = L1ByteStore(num_blocks=16, block_size_bytes=128)
    registry = KVBlockRegistry()
    manager = MultiTieredCacheManager(l1=l1, l2=mock_l2, registry=registry)

    # Use port 0 to let OS pick a free port — but grpc.aio doesn't support it cleanly,
    # so we pick a high port unlikely to collide.
    port = 50099
    server = await create_grpc_server(manager, port=port)

    client = SidecarCacheClient(grpc_url=f"localhost:{port}")
    yield client, manager, registry

    await client.close()
    await server.stop(grace=0)


class TestGRPCStoreLoadRoundtrip:
    @pytest.mark.asyncio
    async def test_store_load_roundtrip(self, grpc_env):
        client, manager, registry = grpc_env

        # Allocate
        ids = await client.allocate_blocks(["grpc-hash-1"])
        assert ids is not None
        assert len(ids) == 1
        block_id = ids[0]

        # Store
        data = b"grpc-test-payload" + b"\x00" * (128 - 17)
        ok = await client.store_block(block_id, "grpc-hash-1", data)
        assert ok is True

        # Load
        loaded = await client.load_block(block_id)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_grpc_allocate_free(self, grpc_env):
        client, *_ = grpc_env

        free_before = await client.get_free_blocks()
        assert free_before == 16

        ids = await client.allocate_blocks(["h1", "h2", "h3"])
        assert ids is not None
        assert len(ids) == 3

        free_after_alloc = await client.get_free_blocks()
        assert free_after_alloc == 13

        for bid in ids:
            ok = await client.free_block(bid)
            assert ok is True

        free_after_free = await client.get_free_blocks()
        assert free_after_free == 16

    @pytest.mark.asyncio
    async def test_load_missing_block(self, grpc_env):
        client, *_ = grpc_env
        data = await client.load_block(9999)
        assert data is None


class TestMockEngineOffloadGRPC:
    @pytest.mark.asyncio
    async def test_mock_engine_offload_fetch(self, grpc_env):
        client, manager, registry = grpc_env

        engine = MockLLMEngine()
        engine.cache_client = client

        # Offload
        data = b"\xAB" * 128
        ok = await engine.offload_block("engine-hash", data, model_id="test-model")
        assert ok is True

        # Verify in registry
        entry = registry.lookup("engine-hash")
        assert entry is not None
        assert entry.model_id == "test-model"

        # Fetch back — we need the block_id that was allocated
        block_id = manager.l1._hash_to_id.get("engine-hash")
        assert block_id is not None

        loaded = await engine.fetch_block("engine-hash", block_id)
        assert loaded == data
