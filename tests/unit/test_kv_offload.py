"""Unit tests for the vLLM OffloadingSpec plugin (no GPU needed)."""

import numpy as np
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
