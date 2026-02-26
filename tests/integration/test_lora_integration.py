"""
Integration tests: full LoRA request lifecycle through the FastAPI engine app
with MockLLMEngine + mocked sidecar HTTP responses.
"""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine.mock_engine import MockLLMEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lora_config():
    """Engine config with LoRA enabled and mock engine."""
    return EngineConfig(
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        adapter_poll_interval=0.01,
        adapter_poll_timeout=5.0,
        enable_engine_mock=True,
    )


@pytest.fixture
def mock_engine(lora_config):
    """MockLLMEngine with LoRA manager that has a mocked sidecar."""
    # We need to patch httpx calls inside LoRAManager before creating the engine
    with patch("data_plane.inference.engine.lora_manager.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()

        async def mock_post(url, **kwargs):
            # Extract adapter name from URL path
            adapter_id = url.split("/adapter/load/")[1]
            return httpx.Response(
                status_code=200,
                json={
                    "status": "loaded",
                    "adapter_identifier": adapter_id,
                    "local_path": f"/mnt/models/{adapter_id.replace('/', '--')}/latest",
                },
                request=httpx.Request("POST", url),
            )

        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        engine = MockLLMEngine(config=lora_config)
        yield engine


@pytest.fixture
def test_client(mock_engine, lora_config):
    """FastAPI TestClient with pre-initialized mock engine."""
    with patch("data_plane.inference.engine.api._engine", mock_engine), \
         patch("data_plane.inference.engine.api._config", lora_config):
        from data_plane.inference.engine.api import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInferenceWithAdapter:
    """POST /inference with adapter_identifier triggers LoRA loading."""

    def test_inference_with_adapter_succeeds(self, test_client, mock_engine):
        response = test_client.post("/inference", json={
            "prompt": "hello",
            "adapter_identifier": "org/my-adapter",
            "adapter_version": "v1",
        })
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["tokens_generated"] > 0

        # Adapter was loaded
        assert mock_engine.lora_load_count == 1
        assert len(mock_engine.loaded_loras) == 1

    def test_inference_without_adapter_works_normally(self, test_client, mock_engine):
        response = test_client.post("/inference", json={
            "prompt": "hello",
        })
        assert response.status_code == 200
        data = response.json()
        assert "text" in data

        # No adapters loaded
        assert mock_engine.lora_load_count == 0
        assert len(mock_engine.loaded_loras) == 0


class TestAdapterCaching:
    """Repeated requests for same adapter don't trigger re-downloads."""

    def test_same_adapter_loaded_once(self, test_client, mock_engine):
        for _ in range(3):
            response = test_client.post("/inference", json={
                "prompt": "test",
                "adapter_identifier": "org/my-adapter",
                "adapter_version": "v1",
            })
            assert response.status_code == 200

        # Only loaded once despite 3 requests
        assert mock_engine.lora_load_count == 1


class TestAdapterEviction:
    """With max_loras=2, loading 3rd adapter triggers eviction."""

    def test_third_adapter_evicts_first(self, test_client, mock_engine):
        adapters = ["org/adapter-a", "org/adapter-b", "org/adapter-c"]
        for adapter in adapters:
            response = test_client.post("/inference", json={
                "prompt": "test",
                "adapter_identifier": adapter,
            })
            assert response.status_code == 200

        # 3 loads, 1 eviction
        assert mock_engine.lora_load_count == 3
        assert mock_engine.lora_remove_count == 1
        assert len(mock_engine.loaded_loras) == 2
