from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    model_config = {"env_prefix": "GATEWAY_"}

    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = 300.0
