from pydantic_settings import BaseSettings

from shared.config_loader import yaml_settings_source


class GatewayConfig(BaseSettings):
    model_config = {"env_prefix": "GATEWAY_"}

    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = 300.0

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],
            kwargs["env_settings"],
            yaml_settings_source("gateway"),
        )
