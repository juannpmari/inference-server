"""
Unit tests for the sidecar API.
Tests health, ready, load, unload, registry, and adapter endpoints
with a mocked ArtifactManager (no real HF downloads).
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_plane.inference.sidecar.artifact_manager import ArtifactManager
from data_plane.inference.sidecar.config import SidecarConfig


@pytest.fixture
def mock_manager():
    """Create a mocked ArtifactManager."""
    manager = MagicMock()
    manager.is_ready = True
    manager.model_registry = {
        "test-model": {
            "model_id": "test-model",
            "version": "v1.0",
            "local_path": "/mnt/models/test-model/v1.0",
            "status": "loaded",
        }
    }
    manager.adapter_registry = {}
    manager.load_model = AsyncMock(return_value="/mnt/models/new-model/latest")
    manager.unload_model = AsyncMock()
    manager.fetch_adapter = AsyncMock(return_value="/mnt/models/adapter/latest")
    return manager


@pytest.fixture
def test_client(mock_manager):
    """Create a FastAPI TestClient with mocked manager."""
    from fastapi.testclient import TestClient

    # Patch global state before importing app
    with patch("data_plane.inference.sidecar.api._manager", mock_manager), patch(
        "data_plane.inference.sidecar.api._config", SidecarConfig()
    ):
        from data_plane.inference.sidecar.api import app

        # Use TestClient without lifespan to avoid real initialization
        client = TestClient(app, raise_server_exceptions=False)
        yield client


class TestSidecarHealth:
    """Tests for health and readiness endpoints."""

    def test_health_returns_ok(self, test_client):
        response = test_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_ready_returns_ready_when_loaded(self, test_client):
        response = test_client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_ready_returns_503_when_not_ready(self, test_client, mock_manager):
        mock_manager.is_ready = False
        response = test_client.get("/readyz")
        assert response.status_code == 503


class TestSidecarModels:
    """Tests for model load/unload endpoints."""

    def test_load_model_new_returns_202(self, test_client, mock_manager):
        response = test_client.post("/load/new-model?version=latest")
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "downloading"
        assert data["model_identifier"] == "new-model"

    def test_load_model_already_loaded_returns_200(self, test_client, mock_manager):
        """Model already in registry as loaded → 200 with local_path."""
        response = test_client.post("/load/test-model?version=v1.0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loaded"
        assert data["local_path"] == "/mnt/models/test-model/v1.0"

    def test_unload_model_success(self, test_client, mock_manager):
        response = test_client.post("/unload/test-model")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_manager.unload_model.assert_called_once_with("test-model")

    def test_unload_model_not_found(self, test_client):
        response = test_client.post("/unload/nonexistent")
        assert response.status_code == 404


class TestSidecarRegistry:
    """Tests for registry query endpoints."""

    def test_get_models(self, test_client, mock_manager):
        response = test_client.get("/registry/models")
        assert response.status_code == 200
        data = response.json()
        assert "test-model" in data
        assert data["test-model"]["status"] == "loaded"

    def test_get_adapters_empty(self, test_client):
        response = test_client.get("/registry/adapters")
        assert response.status_code == 200
        assert response.json() == {}


class TestSidecarAdapters:
    """Tests for adapter load endpoint (fire-and-forget)."""

    def test_load_adapter_new_returns_202(self, test_client, mock_manager):
        response = test_client.post("/adapter/load/test-adapter?version=v1")
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "downloading"
        assert data["adapter_identifier"] == "test-adapter"

    def test_load_adapter_already_loaded_returns_200(self, test_client, mock_manager):
        mock_manager.adapter_registry["test-adapter"] = {
            "adapter_id": "test-adapter",
            "version": "v1",
            "local_path": "/mnt/models/test-adapter/v1",
            "status": "loaded",
        }
        response = test_client.post("/adapter/load/test-adapter?version=v1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loaded"
        assert data["local_path"] == "/mnt/models/test-adapter/v1"

    def test_load_adapter_already_downloading_returns_202(self, test_client, mock_manager):
        mock_manager.adapter_registry["test-adapter"] = {
            "adapter_id": "test-adapter",
            "version": "v1",
            "status": "downloading",
        }
        response = test_client.post("/adapter/load/test-adapter?version=v1")
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "downloading"


class TestSidecarMetrics:
    """Tests for the /metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        # Check for sidecar-specific metrics in output
        body = response.text
        assert "sidecar_resident_models" in body


class TestSidecarConfig:
    """Tests for SidecarConfig."""

    def test_config_defaults(self):
        config = SidecarConfig()
        assert config.port == 8001
        assert config.l1_capacity_mb == 512
        assert config.max_adapters == 10
        assert config.shared_volume == "/mnt/models"

    def test_config_env_override(self, monkeypatch):
        monkeypatch.setenv("SIDECAR_L1_CAPACITY_MB", "1024")
        monkeypatch.setenv("SIDECAR_MAX_ADAPTERS", "5")
        config = SidecarConfig()
        assert config.l1_capacity_mb == 1024
        assert config.max_adapters == 5


# ---------------------------------------------------------------------------
# R3.4 — Secret Management
# ---------------------------------------------------------------------------


@pytest.fixture
def make_manager(tmp_path):
    """Factory to create an ArtifactManager with a tmp shared volume and optional token file."""

    def _factory(hf_token_file=None):
        config = SidecarConfig(
            shared_volume=str(tmp_path / "models"),
            registry_path=str(tmp_path / "registry.json"),
            hf_token_file=hf_token_file,
        )
        return ArtifactManager(config=config)

    return _factory


class TestHfTokenResolution:
    """Tests for HF token resolution logic in ArtifactManager."""

    def test_get_hf_token_from_file(self, tmp_path, make_manager):
        token_file = tmp_path / "token.txt"
        token_file.write_text("hf_abc123\n")
        mgr = make_manager(hf_token_file=str(token_file))
        assert mgr._get_hf_token() == "hf_abc123"

    def test_get_hf_token_from_env(self, monkeypatch, make_manager):
        monkeypatch.setenv("HF_TOKEN", "hf_envtoken")
        mgr = make_manager()
        assert mgr._get_hf_token() == "hf_envtoken"

    def test_get_hf_token_file_takes_precedence(self, tmp_path, monkeypatch, make_manager):
        token_file = tmp_path / "token.txt"
        token_file.write_text("hf_from_file\n")
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        mgr = make_manager(hf_token_file=str(token_file))
        assert mgr._get_hf_token() == "hf_from_file"

    def test_get_hf_token_returns_none(self, monkeypatch, make_manager):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        mgr = make_manager()
        assert mgr._get_hf_token() is None

    def test_startup_raises_on_missing_token_file(self, make_manager):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            make_manager(hf_token_file="/nonexistent/token.txt")

    def test_snapshot_download_receives_token(self, tmp_path, make_manager):
        """Verify that _fetch_from_external_storage passes token= to snapshot_download."""
        token_file = tmp_path / "token.txt"
        token_file.write_text("hf_mytoken\n")
        mgr = make_manager(hf_token_file=str(token_file))

        with patch("data_plane.inference.sidecar.artifact_manager.snapshot_download") as mock_sd:
            mock_sd.return_value = None
            asyncio.get_event_loop().run_until_complete(
                mgr._fetch_from_external_storage("model", "org/model", "main")
            )
            mock_sd.assert_called_once()
            _, kwargs = mock_sd.call_args
            assert kwargs["token"] == "hf_mytoken"


# ---------------------------------------------------------------------------
# R5.4 — Model Download Progress Reporting
# ---------------------------------------------------------------------------


class TestDownloadProgress:
    """Tests for download progress tracking."""

    def test_get_dir_size_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert ArtifactManager._get_dir_size(str(empty_dir)) == 0
        # Nonexistent dir
        assert ArtifactManager._get_dir_size(str(tmp_path / "nonexistent")) == 0

    def test_get_dir_size_with_files(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.bin").write_bytes(b"x" * 100)
        (d / "b.bin").write_bytes(b"y" * 250)
        sub = d / "sub"
        sub.mkdir()
        (sub / "c.bin").write_bytes(b"z" * 50)
        assert ArtifactManager._get_dir_size(str(d)) == 400

    def test_download_progress_entry_lifecycle(self, tmp_path, make_manager):
        """Progress entry exists during download and is removed after."""
        mgr = make_manager()
        captured_progress = {}

        def fake_snapshot_download(**kwargs):
            # During download, progress entry should exist
            captured_progress.update(dict(mgr.download_progress))

        with patch("data_plane.inference.sidecar.artifact_manager.snapshot_download", side_effect=fake_snapshot_download):
            asyncio.get_event_loop().run_until_complete(
                mgr._fetch_from_external_storage("model", "org/model", "main")
            )

        # During download the entry was present
        assert "org/model" in captured_progress
        assert "downloaded_bytes" in captured_progress["org/model"]
        # After download it should be gone
        assert "org/model" not in mgr.download_progress

    def test_status_endpoint_returns_progress(self, test_client, mock_manager):
        """Downloading model with progress data shows download_progress in /status."""
        mock_manager.model_registry["downloading-model"] = {
            "model_id": "downloading-model",
            "version": "main",
            "status": "downloading",
        }
        mock_manager.download_progress = {
            "downloading-model": {
                "downloaded_bytes": 5000,
                "total_bytes": None,
                "started_at": 1000.0,
            }
        }
        response = test_client.get("/status/downloading-model")
        assert response.status_code == 200
        data = response.json()
        assert "download_progress" in data
        assert data["download_progress"]["downloaded_bytes"] == 5000

    def test_status_endpoint_loaded_model_no_progress(self, test_client, mock_manager):
        """Loaded model should not include download_progress."""
        mock_manager.download_progress = {}
        response = test_client.get("/status/test-model")
        assert response.status_code == 200
        data = response.json()
        assert "download_progress" not in data

    def test_status_endpoint_404_unknown_model(self, test_client, mock_manager):
        response = test_client.get("/status/unknown-model")
        assert response.status_code == 404

    def test_registry_models_includes_progress(self, test_client, mock_manager):
        """GET /registry/models merges download_progress for downloading models."""
        mock_manager.model_registry["dl-model"] = {
            "model_id": "dl-model",
            "version": "main",
            "status": "downloading",
        }
        mock_manager.download_progress = {
            "dl-model": {
                "downloaded_bytes": 1234,
                "total_bytes": None,
                "started_at": 999.0,
            }
        }
        response = test_client.get("/registry/models")
        assert response.status_code == 200
        data = response.json()
        assert "dl-model" in data
        assert "download_progress" in data["dl-model"]
        assert data["dl-model"]["download_progress"]["downloaded_bytes"] == 1234
        # Loaded model should NOT have download_progress
        assert "download_progress" not in data["test-model"]
