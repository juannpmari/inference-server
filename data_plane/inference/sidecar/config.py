from pydantic_settings import BaseSettings

from shared.config_loader import get_config


def _sidecar_yaml_source() -> dict:
    """Merge sidecar section with L2 redis settings."""
    cfg = dict(get_config("sidecar"))
    l2 = get_config("l2")
    if l2:
        cfg.setdefault("l2_redis_host", l2.get("redis_host", "localhost"))
        cfg.setdefault("l2_redis_port", l2.get("redis_port", 6379))
    return cfg


class SidecarConfig(BaseSettings):
    model_config = {"env_prefix": "SIDECAR_"}

    host: str = "0.0.0.0"
    port: int = 8001
    shared_volume: str = "/mnt/models"
    max_adapters: int = 10
    l1_capacity_mb: int = 512
    engine_url: str = "http://localhost:8080"
    l2_redis_host: str = "localhost"
    l2_redis_port: int = 6379
    grpc_port: int = 50051
    initial_model: str = "arnir0/Tiny-LLM"
    initial_model_version: str = "main"
    model_store_path: str = "/mnt/models"
    registry_path: str = "/mnt/models/registry.json"
    l1_num_blocks: int = 1024
    l1_block_size_bytes: int = 131072  # 128 KB

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],
            kwargs["env_settings"],
            _sidecar_yaml_source,
        )
