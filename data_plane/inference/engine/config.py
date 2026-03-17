from pydantic import Field
from pydantic_settings import BaseSettings

from shared.config_loader import yaml_settings_source


class EngineConfig(BaseSettings):
    model_config = {
        "env_prefix": "ENGINE_",
        "populate_by_name": True,
    }

    model_name: str = "Qwen/Qwen2-0.5B"
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.70
    dtype: str = "bfloat16"
    host: str = "0.0.0.0"
    port: int = 8080
    sidecar_url: str = "http://localhost:8001"
    sidecar_poll_interval: float = 2.0
    sidecar_timeout: float = 600.0
    model_path: str = "/models/resident_model"
    enable_lora: bool = False
    max_loras: int = 4
    max_lora_rank: int = 16
    adapter_poll_interval: float = 1.0
    adapter_poll_timeout: float = 600.0
    max_pending: int = 10
    temperature: float = 0.0
    sidecar_grpc_url: str = "localhost:50051"
    enable_kv_offload: bool = False
    kv_offload_num_blocks: int = 1024
    enable_engine_mock: bool = Field(
        default=False,
        alias="ENABLE_ENGINE_MOCK",
        description="Set to true to use mock engine (no GPU needed)",
    )

    # GPU monitoring
    gpu_monitor_enabled: bool = True
    gpu_poll_interval: float = 2.0
    gpu_device_index: int = 0

    # Metrics persistence
    monitoring_storage_backend: str = "local"
    monitoring_local_store_path: str = "/mnt/models/metrics.jsonl"
    monitoring_buffer_size: int = 10000
    monitoring_flush_interval: float = 30.0

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],    # explicit kwargs (highest)
            kwargs["env_settings"],     # environment variables
            yaml_settings_source("engine"),  # YAML config file
        )
