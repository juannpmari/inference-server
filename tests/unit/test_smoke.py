import shared.ports
import shared.types
from data_plane.inference.engine.config import EngineConfig


def test_engine_config_defaults(engine_config):
    assert engine_config.model_name == "Qwen/Qwen2-0.5B"
    assert engine_config.port == 8080
    assert engine_config.gpu_memory_utilization == 0.70


def test_sidecar_config_defaults(sidecar_config):
    assert sidecar_config.l1_capacity_mb == 512
    assert sidecar_config.port == 8001
    assert sidecar_config.max_adapters == 10


def test_gateway_config_defaults(gateway_config):
    assert gateway_config.port == 8000
    assert gateway_config.request_timeout == 300.0


def test_engine_env_override(monkeypatch):
    monkeypatch.setenv("ENGINE_MODEL_NAME", "test-model")
    cfg = EngineConfig()
    assert cfg.model_name == "test-model"


def test_shared_imports():
    assert hasattr(shared.types, "BlockReference")
    assert hasattr(shared.types, "TransferResult")
    assert hasattr(shared.ports, "CacheStore")
    assert hasattr(shared.ports, "ModelRepository")
