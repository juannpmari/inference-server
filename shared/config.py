"""Unified configuration system for the inference server.

Provides a single ``InferenceServerConfig`` that aggregates gateway, engine,
and sidecar configuration into one validated model.  Each section can still
be used independently via its own class (``GatewaySection``, ``EngineSection``,
``SidecarSection``) — the per-service ``GatewayConfig`` / ``EngineConfig`` /
``SidecarConfig`` thin wrappers delegate here.

Loading priority (highest → lowest):
  1. ``GATEWAY_*`` / ``ENGINE_*`` / ``SIDECAR_*`` environment-variable overrides
  2. YAML config file (``INFERENCE_SERVER_CONFIG`` → ``SERVER_CONFIG_PATH`` →
     ``server_config.yaml``)
  3. Class-level defaults
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "server_config.yaml"


@lru_cache(maxsize=1)
def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        logger.warning("Config file not found at %s, using empty config", p)
        return {}
    with open(p) as f:
        data = yaml.safe_load(f)
    logger.info("Loaded server config from %s", p)
    return data or {}


def _resolve_config_path() -> str:
    """Determine which YAML config file to use."""
    for var in ("INFERENCE_SERVER_CONFIG", "SERVER_CONFIG_PATH"):
        val = os.environ.get(var)
        if val:
            return val
    return str(_DEFAULT_CONFIG_PATH)


def _get_section(section: str) -> dict[str, Any]:
    """Return a top-level section from the YAML config."""
    path = _resolve_config_path()
    cfg = _load_yaml(path)
    return cfg.get(section, {})


def _env_overlay(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    """Apply ``PREFIX_FIELD`` env-var overrides onto *data*."""
    result = dict(data)
    for key, val in os.environ.items():
        if key.startswith(prefix):
            field = key[len(prefix):].lower()
            result[field] = val
    return result


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class GatewaySection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = 300.0
    drain_timeout: float = 30.0
    routes: Dict[str, str] = Field(default_factory=dict)
    log_json: bool = True
    log_level: str = "INFO"
    rate_limit_rps: float = 100.0
    rate_limit_burst: int = 200
    max_request_body_bytes: int = 1_048_576  # 1 MB


class EngineSection(BaseModel):
    model_name: str = "Qwen/Qwen2-0.5B"
    max_model_len: int = 4000
    gpu_memory_utilization: float = 0.70
    dtype: str = "bfloat16"
    host: str = "0.0.0.0"
    port: int = 8080
    sidecar_url: str = "http://localhost:8001"
    sidecar_poll_interval: float = 2.0
    sidecar_timeout: float = 280.0
    model_path: str = "/models/resident_model"
    enable_lora: bool = False
    max_loras: int = 4
    max_lora_rank: int = 16
    adapter_poll_interval: float = 1.0
    adapter_poll_timeout: float = 600.0
    max_pending: int = 200
    temperature: float = 0.0
    sidecar_grpc_url: str = "localhost:50051"
    enable_prefix_caching: bool = False
    enable_kv_offload: bool = False
    kv_offload_num_blocks: int = 1024
    enable_engine_mock: bool = Field(
        default=False,
        description="Set to true to use mock engine (no GPU needed)",
    )
    # Graceful shutdown
    drain_timeout: float = 30.0
    # GPU monitoring
    gpu_monitor_enabled: bool = True
    gpu_poll_interval: float = 2.0
    gpu_device_index: int = 0
    # Metrics persistence
    monitoring_storage_backend: str = "local"
    monitoring_local_store_path: str = "/mnt/models/metrics.jsonl"
    monitoring_buffer_size: int = 10000
    monitoring_flush_interval: float = 30.0
    # New: inference timeout (used by generate endpoint).
    # Must be < sidecar_timeout so inference completes before sidecar poll gives up.
    inference_timeout: float = 270.0
    log_json: bool = True
    log_level: str = "INFO"


class SidecarSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    shared_volume: str = "/mnt/models"
    max_adapters: int = 10
    l1_capacity_mb: int = 512
    engine_url: str = "http://localhost:8080"
    l2_redis_host: str = "localhost"
    l2_redis_port: int = 6379
    grpc_port: int = 50051
    initial_model: str = "Qwen/Qwen2-0.5B"
    initial_model_version: str = "main"
    model_store_path: str = "/mnt/models"
    registry_path: str = "/mnt/models/registry.json"
    l1_num_blocks: int = 1024
    l1_block_size_bytes: int = 131072  # 128 KB
    # New fields
    hf_token_file: Optional[str] = None
    verify_checksums: bool = True
    log_json: bool = True
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Aggregate config
# ---------------------------------------------------------------------------

class InferenceServerConfig(BaseModel):
    gateway: GatewaySection = Field(default_factory=GatewaySection)
    engine: EngineSection = Field(default_factory=EngineSection)
    sidecar: SidecarSection = Field(default_factory=SidecarSection)

    @model_validator(mode="after")
    def _cross_component_checks(self) -> "InferenceServerConfig":
        # Warn if model names don't match
        if self.engine.model_name != self.sidecar.initial_model:
            logger.warning(
                "engine.model_name (%s) != sidecar.initial_model (%s) — "
                "the sidecar may download a different model than the engine expects",
                self.engine.model_name,
                self.sidecar.initial_model,
            )
        # Error if gateway timeout is not larger than engine inference timeout
        if self.gateway.request_timeout <= self.engine.inference_timeout:
            raise ValueError(
                f"gateway.request_timeout ({self.gateway.request_timeout}) must be "
                f"greater than engine.inference_timeout ({self.engine.inference_timeout})"
            )
        # Timeout hierarchy: engine.sidecar_timeout must fit within
        # gateway.request_timeout with a 10s buffer for gateway overhead.
        if self.engine.sidecar_timeout >= (self.gateway.request_timeout - 10.0):
            raise ValueError(
                f"engine.sidecar_timeout ({self.engine.sidecar_timeout}) must be "
                f"less than gateway.request_timeout - 10s "
                f"({self.gateway.request_timeout - 10.0})"
            )
        # engine.inference_timeout must be less than engine.sidecar_timeout
        # so that inference completes before the sidecar poll gives up.
        if self.engine.inference_timeout >= self.engine.sidecar_timeout:
            raise ValueError(
                f"engine.inference_timeout ({self.engine.inference_timeout}) must be "
                f"less than engine.sidecar_timeout ({self.engine.sidecar_timeout})"
            )
        return self


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config() -> InferenceServerConfig:
    """Build the unified config from YAML + env overlays."""
    path = _resolve_config_path()
    raw = _load_yaml(path)

    gateway_data = _env_overlay("GATEWAY_", dict(raw.get("gateway", {})))
    engine_data = _env_overlay("ENGINE_", dict(raw.get("engine", {})))
    sidecar_data = _env_overlay("SIDECAR_", dict(raw.get("sidecar", {})))

    # Merge L2 redis settings into sidecar (preserving existing behaviour)
    l2 = raw.get("l2", {})
    if l2:
        sidecar_data.setdefault("l2_redis_host", l2.get("redis_host", "localhost"))
        sidecar_data.setdefault("l2_redis_port", l2.get("redis_port", 6379))

    # Populate gateway routes from routing.model_service_map
    routing = raw.get("routing", {})
    if routing and "routes" not in gateway_data:
        gateway_data["routes"] = routing.get("model_service_map", {})

    return InferenceServerConfig(
        gateway=GatewaySection(**gateway_data),
        engine=EngineSection(**engine_data),
        sidecar=SidecarSection(**sidecar_data),
    )
