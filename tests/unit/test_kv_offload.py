"""Unit tests for the vLLM OffloadingSpec plugin (no GPU needed)."""

import pytest

from data_plane.inference.engine.kv_offload.sidecar_backend import (
    SidecarBackend,
    SidecarLoadStoreSpec,
)
from data_plane.inference.engine.kv_offload.sidecar_spec import SidecarOffloadingSpec


class TestSidecarBackend:
    def test_allocate_free(self):
        backend = SidecarBackend(num_blocks=8)
        assert backend.get_num_free_blocks() == 8

        ids = backend.allocate_blocks(["h1", "h2", "h3"])
        assert ids is not None
        assert len(ids) == 3
        assert backend.get_num_free_blocks() == 5

        backend.free_block(ids[0])
        assert backend.get_num_free_blocks() == 6

    def test_allocate_exceeds_capacity(self):
        backend = SidecarBackend(num_blocks=2)
        ids = backend.allocate_blocks(["a", "b", "c"])
        assert ids is None  # Only 2 blocks available

    def test_load_store_spec(self):
        backend = SidecarBackend(num_blocks=4)
        ids = backend.allocate_blocks(["h1", "h2"])
        assert ids is not None

        spec = backend.get_load_store_spec(ids, ["h1", "h2"])
        assert isinstance(spec, SidecarLoadStoreSpec)
        assert spec.block_ids == ids
        assert spec.block_hashes == ["h1", "h2"]

    def test_load_store_spec_medium(self):
        spec = SidecarLoadStoreSpec(block_ids=[1], block_hashes=["h"])
        assert spec.medium() == "SIDECAR"

    def test_duplicate_hash_returns_existing_id(self):
        backend = SidecarBackend(num_blocks=4)
        ids1 = backend.allocate_blocks(["h1"])
        ids2 = backend.allocate_blocks(["h1"])  # same hash
        assert ids1 == ids2  # same block_id returned


class TestSidecarOffloadingSpec:
    def test_default_config(self):
        spec = SidecarOffloadingSpec(vllm_config=None)
        assert spec._grpc_url == "localhost:50051"
        assert spec._num_blocks == 1024
        assert spec.backend is not None
        assert spec.client is not None

    def test_custom_config(self):
        class FakeKVConfig:
            kv_connector_extra_config = {
                "sidecar_grpc_url": "localhost:9999",
                "num_blocks": 512,
                "block_size_bytes": 65536,
            }

        class FakeVllmConfig:
            kv_transfer_config = FakeKVConfig()

        spec = SidecarOffloadingSpec(vllm_config=FakeVllmConfig())
        assert spec._grpc_url == "localhost:9999"
        assert spec._num_blocks == 512
        assert spec._block_size == 65536

    def test_backend_tracks_blocks(self):
        spec = SidecarOffloadingSpec(vllm_config=None)
        assert spec.backend.get_num_free_blocks() == 1024

        ids = spec.backend.allocate_blocks(["x", "y"])
        assert ids is not None
        assert spec.backend.get_num_free_blocks() == 1022
