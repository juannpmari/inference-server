"""Engine configuration — thin wrapper delegating to ``shared.config``.

Backward-compatible: ``EngineConfig()`` still works as before, but field
definitions now come from the unified ``EngineSection`` model.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from shared.config import EngineSection, _get_section, _env_overlay


def _engine_yaml_source() -> dict:
    data = _get_section("engine")
    return _env_overlay("ENGINE_", data)


class EngineConfig(BaseSettings):
    """Engine settings — mirrors ``shared.config.EngineSection``."""

    model_config = {
        "env_prefix": "ENGINE_",
        "populate_by_name": True,
    }

    model_name: str = EngineSection.model_fields["model_name"].default
    max_model_len: int = EngineSection.model_fields["max_model_len"].default
    gpu_memory_utilization: float = EngineSection.model_fields["gpu_memory_utilization"].default
    dtype: str = EngineSection.model_fields["dtype"].default
    host: str = EngineSection.model_fields["host"].default
    port: int = EngineSection.model_fields["port"].default
    sidecar_url: str = EngineSection.model_fields["sidecar_url"].default
    sidecar_poll_interval: float = EngineSection.model_fields["sidecar_poll_interval"].default
    sidecar_timeout: float = EngineSection.model_fields["sidecar_timeout"].default
    model_path: str = EngineSection.model_fields["model_path"].default
    enable_lora: bool = EngineSection.model_fields["enable_lora"].default
    max_loras: int = EngineSection.model_fields["max_loras"].default
    max_lora_rank: int = EngineSection.model_fields["max_lora_rank"].default
    adapter_poll_interval: float = EngineSection.model_fields["adapter_poll_interval"].default
    adapter_poll_timeout: float = EngineSection.model_fields["adapter_poll_timeout"].default
    max_pending: int = EngineSection.model_fields["max_pending"].default
    temperature: float = EngineSection.model_fields["temperature"].default
    sidecar_grpc_url: str = EngineSection.model_fields["sidecar_grpc_url"].default
    enable_prefix_caching: bool = EngineSection.model_fields["enable_prefix_caching"].default
    enable_kv_offload: bool = EngineSection.model_fields["enable_kv_offload"].default
    kv_offload_num_blocks: int = EngineSection.model_fields["kv_offload_num_blocks"].default
    enable_engine_mock: bool = Field(
        default=False,
        alias="ENABLE_ENGINE_MOCK",
        description="Set to true to use mock engine (no GPU needed)",
    )

    # Graceful shutdown
    drain_timeout: float = EngineSection.model_fields["drain_timeout"].default

    # GPU monitoring
    gpu_monitor_enabled: bool = EngineSection.model_fields["gpu_monitor_enabled"].default
    gpu_poll_interval: float = EngineSection.model_fields["gpu_poll_interval"].default
    gpu_device_index: int = EngineSection.model_fields["gpu_device_index"].default

    # Metrics persistence
    monitoring_storage_backend: str = EngineSection.model_fields["monitoring_storage_backend"].default
    monitoring_local_store_path: str = EngineSection.model_fields["monitoring_local_store_path"].default
    monitoring_buffer_size: int = EngineSection.model_fields["monitoring_buffer_size"].default
    monitoring_flush_interval: float = EngineSection.model_fields["monitoring_flush_interval"].default

    # Inference timeout (replaces sidecar_timeout for generate endpoint)
    inference_timeout: float = EngineSection.model_fields["inference_timeout"].default

    log_json: bool = EngineSection.model_fields["log_json"].default
    log_level: str = EngineSection.model_fields["log_level"].default

    otlp_endpoint: Optional[str] = None

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],    # explicit kwargs (highest)
            kwargs["env_settings"],     # environment variables
            _engine_yaml_source,        # YAML config file
        )
