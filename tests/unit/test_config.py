"""Tests for shared.config — timeout validation, YAML loading, env overlays."""

import os
import textwrap

import pytest

from shared.config import (
    GatewaySection,
    EngineSection,
    SidecarSection,
    InferenceServerConfig,
    _load_yaml,
    _env_overlay,
    _resolve_config_path,
    load_config,
)


class TestTimeoutValidation:

    def test_valid_timeout_hierarchy(self):
        """Default values should pass all cross-component checks."""
        config = InferenceServerConfig(
            gateway=GatewaySection(request_timeout=300.0),
            engine=EngineSection(sidecar_timeout=280.0, inference_timeout=270.0),
            sidecar=SidecarSection(),
        )
        assert config.gateway.request_timeout == 300.0

    def test_gateway_timeout_must_exceed_inference_timeout(self):
        with pytest.raises(ValueError, match="gateway.request_timeout"):
            InferenceServerConfig(
                gateway=GatewaySection(request_timeout=100.0),
                engine=EngineSection(sidecar_timeout=80.0, inference_timeout=100.0),
                sidecar=SidecarSection(),
            )

    def test_sidecar_timeout_must_fit_within_gateway_timeout(self):
        """engine.sidecar_timeout must be < gateway.request_timeout - 10s."""
        with pytest.raises(ValueError, match="engine.sidecar_timeout"):
            InferenceServerConfig(
                gateway=GatewaySection(request_timeout=300.0),
                engine=EngineSection(sidecar_timeout=295.0, inference_timeout=280.0),
                sidecar=SidecarSection(),
            )

    def test_inference_timeout_must_be_less_than_sidecar_timeout(self):
        """engine.inference_timeout must be < engine.sidecar_timeout."""
        with pytest.raises(ValueError, match="engine.inference_timeout"):
            InferenceServerConfig(
                gateway=GatewaySection(request_timeout=300.0),
                engine=EngineSection(sidecar_timeout=200.0, inference_timeout=200.0),
                sidecar=SidecarSection(),
            )

    def test_defaults_pass_validation(self):
        """Using all default values should produce a valid config."""
        config = InferenceServerConfig()
        assert config.engine.sidecar_timeout < config.gateway.request_timeout - 10.0
        assert config.engine.inference_timeout < config.engine.sidecar_timeout

    def test_rate_limit_defaults(self):
        """GatewaySection should have rate limit fields."""
        gs = GatewaySection()
        assert gs.rate_limit_rps == 100.0
        assert gs.rate_limit_burst == 200

    def test_model_name_mismatch_warns(self, caplog):
        """Mismatched model names should log a warning but not raise."""
        with caplog.at_level("WARNING"):
            InferenceServerConfig(
                gateway=GatewaySection(),
                engine=EngineSection(model_name="modelA"),
                sidecar=SidecarSection(initial_model="modelB"),
            )
        assert "modelA" in caplog.text
        assert "modelB" in caplog.text


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYAMLLoading:

    def test_load_yaml_from_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            gateway:
              port: 9000
            engine:
              model_name: test-model
        """))
        _load_yaml.cache_clear()
        data = _load_yaml(str(cfg_file))
        assert data["gateway"]["port"] == 9000
        assert data["engine"]["model_name"] == "test-model"

    def test_load_yaml_missing_file_returns_empty(self, tmp_path):
        _load_yaml.cache_clear()
        data = _load_yaml(str(tmp_path / "nonexistent.yaml"))
        assert data == {}

    def test_load_yaml_empty_file_returns_empty(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        _load_yaml.cache_clear()
        data = _load_yaml(str(cfg_file))
        assert data == {}


# ---------------------------------------------------------------------------
# Environment variable overlays
# ---------------------------------------------------------------------------

class TestEnvOverlay:

    def test_env_overlay_applies_prefix(self):
        env = {"GATEWAY_PORT": "9090", "GATEWAY_LOG_LEVEL": "DEBUG", "OTHER_VAR": "x"}
        with pytest.MonkeyPatch.context() as mp:
            for k, v in env.items():
                mp.setenv(k, v)
            result = _env_overlay("GATEWAY_", {"port": 8000})
        assert result["port"] == "9090"
        assert result["log_level"] == "DEBUG"
        assert "other_var" not in result

    def test_env_overlay_overrides_existing(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("ENGINE_MODEL_NAME", "override-model")
            result = _env_overlay("ENGINE_", {"model_name": "original"})
        assert result["model_name"] == "override-model"


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------

class TestConfigPathResolution:

    def test_inference_server_config_env_takes_priority(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_SERVER_CONFIG", "/custom/path.yaml")
        monkeypatch.delenv("SERVER_CONFIG_PATH", raising=False)
        assert _resolve_config_path() == "/custom/path.yaml"

    def test_server_config_path_fallback(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_SERVER_CONFIG", raising=False)
        monkeypatch.setenv("SERVER_CONFIG_PATH", "/other/path.yaml")
        assert _resolve_config_path() == "/other/path.yaml"

    def test_defaults_to_server_config_yaml(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_SERVER_CONFIG", raising=False)
        monkeypatch.delenv("SERVER_CONFIG_PATH", raising=False)
        path = _resolve_config_path()
        assert path.endswith("server_config.yaml")


# ---------------------------------------------------------------------------
# load_config integration
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_load_config_from_yaml_with_env_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            gateway:
              port: 9000
              request_timeout: 300.0
            engine:
              model_name: yaml-model
              sidecar_timeout: 280.0
              inference_timeout: 270.0
            sidecar:
              initial_model: yaml-model
        """))
        _load_yaml.cache_clear()
        monkeypatch.setenv("INFERENCE_SERVER_CONFIG", str(cfg_file))
        # Override engine model name via env
        monkeypatch.setenv("ENGINE_MODEL_NAME", "env-model")
        monkeypatch.setenv("SIDECAR_INITIAL_MODEL", "env-model")

        config = load_config()
        assert config.gateway.port == 9000
        assert config.engine.model_name == "env-model"

    def test_load_config_l2_redis_backward_compat(self, tmp_path, monkeypatch):
        """Legacy l2 section should populate sidecar redis settings."""
        cfg_file = tmp_path / "test_l2.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            l2:
              redis_host: redis.internal
              redis_port: 6380
            sidecar:
              initial_model: Qwen/Qwen2-0.5B
        """))
        _load_yaml.cache_clear()
        monkeypatch.setenv("INFERENCE_SERVER_CONFIG", str(cfg_file))

        config = load_config()
        assert config.sidecar.l2_redis_host == "redis.internal"
        assert config.sidecar.l2_redis_port == 6380
