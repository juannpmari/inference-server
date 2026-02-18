"""
Unit tests for the inference engine.
Tests both real and mock engine implementations.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine.mock_engine import MockLLMEngine
from data_plane.inference.engine.api import InferenceRequest, app


@pytest.fixture
def engine_config():
    """Create a test engine config"""
    return EngineConfig(enable_engine_mock=True)


@pytest.fixture
def mock_engine(engine_config):
    """Create a mock engine instance"""
    return MockLLMEngine(engine_config)


class TestMockEngine:
    """Tests for the MockLLMEngine"""

    @pytest.mark.asyncio
    async def test_mock_engine_initialization(self, mock_engine):
        """Test that mock engine initializes correctly"""
        assert mock_engine is not None
        assert mock_engine.is_ready()

    @pytest.mark.asyncio
    async def test_mock_engine_add_request(self, mock_engine):
        """Test adding a request to mock engine"""
        result = await mock_engine.add_request(prompt="hello world")
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_mock_engine_deterministic_responses(self, mock_engine):
        """Test that mock engine returns deterministic responses"""
        result1 = await mock_engine.add_request(prompt="hello")
        result2 = await mock_engine.add_request(prompt="hello")
        # Both should match the deterministic response for "hello"
        assert "hello" in result1.lower() or "help" in result1.lower()
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_mock_engine_has_unfinished_requests(self, mock_engine):
        """Test pending requests tracking"""
        assert not mock_engine.has_unfinished_requests()

        # Add request
        mock_engine.pending_requests["1"] = {"prompt": "test", "response": "response"}
        assert mock_engine.has_unfinished_requests()

        # Clear requests
        await mock_engine.step()
        assert not mock_engine.has_unfinished_requests()

    @pytest.mark.asyncio
    async def test_mock_engine_step_returns_outputs(self, mock_engine):
        """Test the step method returns correct output format"""
        mock_engine.pending_requests["1"] = {
            "prompt": "test",
            "response": "test response"
        }

        outputs = await mock_engine.step()

        assert len(outputs) == 1
        assert outputs[0].request_id == "1"
        assert outputs[0].finished
        assert outputs[0].outputs[0].text == "test response"

    @pytest.mark.asyncio
    async def test_mock_engine_batching_loop(self, mock_engine):
        """Test the batching loop handles cancellation"""
        loop_task = asyncio.create_task(mock_engine.continuous_batching_loop())

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Cancel the loop
        loop_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await loop_task


class TestEngineConfig:
    """Tests for EngineConfig"""

    def test_config_defaults(self):
        """Test config has correct defaults"""
        config = EngineConfig()
        assert config.model_name == "Qwen/Qwen2-0.5B"
        assert config.max_model_len == 512
        assert config.enable_engine_mock == False

    def test_config_enable_mock_flag(self):
        """Test mock flag can be set"""
        config = EngineConfig(enable_engine_mock=True)
        assert config.enable_engine_mock == True

    def test_config_from_env(self, monkeypatch):
        """Test config can be loaded from environment"""
        monkeypatch.setenv("ENGINE_MODEL_NAME", "test-model")
        monkeypatch.setenv("ENGINE_ENABLE_ENGINE_MOCK", "true")

        config = EngineConfig()
        assert config.model_name == "test-model"
        assert config.enable_engine_mock == True


class TestInferenceRequest:
    """Tests for the InferenceRequest model"""

    def test_valid_request(self):
        """Test creating valid inference request"""
        req = InferenceRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.stream == False

    def test_request_validates_prompt(self):
        """Test prompt validation"""
        with pytest.raises(ValueError):
            InferenceRequest(prompt="")  # Empty prompt should fail

    def test_request_validates_max_tokens(self):
        """Test max_tokens validation"""
        with pytest.raises(ValueError):
            InferenceRequest(prompt="hello", max_tokens=0)  # Too low

        with pytest.raises(ValueError):
            InferenceRequest(prompt="hello", max_tokens=5000)  # Too high

    def test_request_validates_temperature(self):
        """Test temperature validation"""
        with pytest.raises(ValueError):
            InferenceRequest(prompt="hello", temperature=-0.1)  # Too low

        with pytest.raises(ValueError):
            InferenceRequest(prompt="hello", temperature=2.1)  # Too high

    def test_request_custom_values(self):
        """Test request with custom values"""
        req = InferenceRequest(
            prompt="test",
            max_tokens=512,
            temperature=0.5,
            stream=True,
            adapter_identifier="test-adapter"
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.5
        assert req.stream == True
        assert req.adapter_identifier == "test-adapter"


class TestEngineAPI:
    """Tests for the FastAPI endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test /health endpoint"""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        # Should return 200 or 503 depending on engine state
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_ready_endpoint(self):
        """Test /ready endpoint"""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/ready")

        # Should return 200 or 503 depending on engine state
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        # Check for Prometheus format (text/plain)
        assert "text/plain" in response.headers.get("content-type", "")


class TestMockEngineResponses:
    """Tests for mock engine response generation"""

    def test_mock_response_for_hello(self):
        """Test mock response for 'hello' prompt"""
        from data_plane.inference.engine.mock_engine import MockLLMEngine

        response = MockLLMEngine._generate_mock_response("hello world")
        assert "hello" in response.lower() or "help" in response.lower()

    def test_mock_response_for_who(self):
        """Test mock response for 'who' prompt"""
        from data_plane.inference.engine.mock_engine import MockLLMEngine

        response = MockLLMEngine._generate_mock_response("Who are you?")
        assert "ai" in response.lower()

    def test_mock_default_response(self):
        """Test mock returns default response for unknown prompt"""
        from data_plane.inference.engine.mock_engine import MockLLMEngine

        response = MockLLMEngine._generate_mock_response("xyzabc")
        assert "quick brown fox" in response.lower()
