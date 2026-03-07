"""
Unit tests for registry persistence (Phase H).
Tests JSON round-trip, metadata with tags/warmup/device, and registry query endpoints.
"""

import asyncio
import json
import os
import tempfile

import pytest

from data_plane.inference.sidecar.artifact_manager import ArtifactManager
from data_plane.inference.sidecar.config import SidecarConfig


class TestRegistryPersistence:
    """Tests for JSON file persistence round-trip."""

    def test_persist_and_restore_model_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)

            am1 = ArtifactManager(config=config)
            am1.model_registry["test-model"] = {
                "model_id": "test-model",
                "local_path": "/tmp/test",
                "status": "loaded",
            }
            am1._persist_registry()

            # File should exist with valid JSON
            assert os.path.exists(registry_file)
            with open(registry_file) as f:
                data = json.load(f)
            assert "models" in data
            assert "test-model" in data["models"]

            # New instance should restore
            am2 = ArtifactManager(config=config)
            assert "test-model" in am2.model_registry
            assert am2.model_registry["test-model"]["status"] == "loaded"

    def test_persist_and_restore_adapter_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)

            am1 = ArtifactManager(config=config)
            am1.adapter_registry["test-adapter"] = {
                "adapter_id": "test-adapter",
                "local_path": "/tmp/adapter",
                "status": "loaded",
            }
            am1._persist_registry()

            am2 = ArtifactManager(config=config)
            assert "test-adapter" in am2.adapter_registry

    def test_empty_registry_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)

            am = ArtifactManager(config=config)
            assert am.model_registry == {}
            assert am.adapter_registry == {}
            assert not os.path.exists(registry_file)

    def test_corrupt_registry_file_handled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            with open(registry_file, "w") as f:
                f.write("not valid json{{{")

            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)
            am = ArtifactManager(config=config)
            # Should gracefully handle corruption
            assert am.model_registry == {}
            assert am.adapter_registry == {}


class TestRegistryMetadata:
    """Tests for tags, warmup prompts, and preferred device metadata (Task 6.3)."""

    def test_registry_entry_with_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)
            am = ArtifactManager(config=config)

            am.model_registry["tagged-model"] = {
                "model_id": "tagged-model",
                "local_path": "/tmp/test",
                "status": "loaded",
                "tags": ["production", "v2"],
            }
            am._persist_registry()

            am2 = ArtifactManager(config=config)
            entry = am2.model_registry["tagged-model"]
            assert entry["tags"] == ["production", "v2"]

    def test_registry_entry_with_warmup_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)
            am = ArtifactManager(config=config)

            am.model_registry["warmup-model"] = {
                "model_id": "warmup-model",
                "local_path": "/tmp/test",
                "status": "loaded",
                "warmup_prompts": ["Hello", "What is 2+2?"],
            }
            am._persist_registry()

            am2 = ArtifactManager(config=config)
            entry = am2.model_registry["warmup-model"]
            assert entry["warmup_prompts"] == ["Hello", "What is 2+2?"]

    def test_registry_entry_with_preferred_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)
            am = ArtifactManager(config=config)

            am.model_registry["gpu-model"] = {
                "model_id": "gpu-model",
                "local_path": "/tmp/test",
                "status": "loaded",
                "preferred_device": "cuda:0",
            }
            am._persist_registry()

            am2 = ArtifactManager(config=config)
            entry = am2.model_registry["gpu-model"]
            assert entry["preferred_device"] == "cuda:0"

    def test_registry_entry_with_all_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_file = os.path.join(tmpdir, "registry.json")
            config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir, shared_volume=tmpdir)
            am = ArtifactManager(config=config)

            am.model_registry["full-model"] = {
                "model_id": "full-model",
                "local_path": "/tmp/test",
                "status": "loaded",
                "tags": ["staging"],
                "warmup_prompts": ["Hi there"],
                "preferred_device": "cuda:1",
            }
            am._persist_registry()

            am2 = ArtifactManager(config=config)
            entry = am2.model_registry["full-model"]
            assert entry["tags"] == ["staging"]
            assert entry["warmup_prompts"] == ["Hi there"]
            assert entry["preferred_device"] == "cuda:1"
