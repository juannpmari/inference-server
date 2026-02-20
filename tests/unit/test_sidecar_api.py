"""
Unit tests for the sidecar API.
Tests health, ready, load, unload, registry, and adapter endpoints
with a mocked ArtifactManager (no real HF downloads).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    manager.unload_model = MagicMock()
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
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_returns_ready_when_loaded(self, test_client):
        response = test_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "test-model" in data["resident_models"]

    def test_ready_returns_503_when_not_ready(self, test_client, mock_manager):
        mock_manager.is_ready = False
        response = test_client.get("/ready")
        assert response.status_code == 503


class TestSidecarModels:
    """Tests for model load/unload endpoints."""

    def test_load_model_success(self, test_client, mock_manager):
        response = test_client.post("/load/new-model?version=latest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_identifier"] == "new-model"
        assert data["local_path"] == "/mnt/models/new-model/latest"
        mock_manager.load_model.assert_called_once_with("new-model", "latest")

    def test_load_model_failure(self, test_client, mock_manager):
        mock_manager.load_model = AsyncMock(side_effect=RuntimeError("Download failed"))
        response = test_client.post("/load/bad-model?version=v1")
        assert response.status_code == 500
        assert "Failed to load model" in response.json()["detail"]

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
    """Tests for adapter fetch endpoint."""

    def test_fetch_adapter_success(self, test_client, mock_manager):
        response = test_client.post("/adapter/fetch/test-adapter?version=v1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["adapter_identifier"] == "test-adapter"
        assert data["local_path"] == "/mnt/models/adapter/latest"
        mock_manager.fetch_adapter.assert_called_once_with("test-adapter", "v1")

    def test_fetch_adapter_failure(self, test_client, mock_manager):
        mock_manager.fetch_adapter = AsyncMock(side_effect=RuntimeError("Not found"))
        response = test_client.post("/adapter/fetch/bad-adapter?version=v1")
        assert response.status_code == 500
        assert "Failed to fetch adapter" in response.json()["detail"]


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
