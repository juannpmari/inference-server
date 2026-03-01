from pydantic_settings import BaseSettings


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
    registry_path: str = "/mnt/models/registry.json"
