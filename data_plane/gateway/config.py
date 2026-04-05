"""Gateway configuration — thin wrapper delegating to ``shared.config``.

Backward-compatible: ``GatewayConfig()`` still works as before, but fields
now come from the unified ``GatewaySection`` model.
"""

from typing import Optional

from pydantic_settings import BaseSettings

from shared.config import GatewaySection, _get_section, _env_overlay


def _gateway_yaml_source() -> dict:
    data = _get_section("gateway")
    return _env_overlay("GATEWAY_", data)


class GatewayConfig(BaseSettings):
    """Gateway settings — mirrors ``shared.config.GatewaySection``."""

    model_config = {"env_prefix": "GATEWAY_"}

    host: str = GatewaySection.model_fields["host"].default
    port: int = GatewaySection.model_fields["port"].default
    request_timeout: float = GatewaySection.model_fields["request_timeout"].default
    drain_timeout: float = GatewaySection.model_fields["drain_timeout"].default
    routes: dict = GatewaySection.model_fields["routes"].default_factory()  # type: ignore[misc]
    log_json: bool = GatewaySection.model_fields["log_json"].default
    log_level: str = GatewaySection.model_fields["log_level"].default
    rate_limit_rps: float = GatewaySection.model_fields["rate_limit_rps"].default
    rate_limit_burst: int = GatewaySection.model_fields["rate_limit_burst"].default
    max_request_body_bytes: int = GatewaySection.model_fields["max_request_body_bytes"].default
    otlp_endpoint: Optional[str] = None

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],
            kwargs["env_settings"],
            _gateway_yaml_source,
        )
