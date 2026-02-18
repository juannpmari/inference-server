from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class EngineConfig(BaseSettings):
    model_config = {"env_prefix": "ENGINE_"}

    model_name: str = "Qwen/Qwen2-0.5B"
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.70
    dtype: str = "bfloat16"
    host: str = "0.0.0.0"
    port: int = 8080
    sidecar_url: str = "http://localhost:8001"
    model_path: str = "/models/resident_model"
    enable_lora: bool = False
    max_pending: int = 10
    temperature: float = 0.0
    enable_engine_mock: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_ENGINE_MOCK", "ENGINE_ENABLE_ENGINE_MOCK"),
        description="Set to true to use mock engine (no GPU needed)"
    )
