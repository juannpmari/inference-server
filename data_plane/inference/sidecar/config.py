"""Sidecar configuration — thin wrapper delegating to ``shared.config``.

Backward-compatible: ``SidecarConfig()`` still works as before, but field
definitions now come from the unified ``SidecarSection`` model.
"""

from typing import Optional

from pydantic_settings import BaseSettings

from shared.config import SidecarSection, _get_section, _env_overlay
from shared.config_loader import get_config


def _sidecar_yaml_source() -> dict:
    """Merge sidecar section with L2 redis settings."""
    cfg = dict(_get_section("sidecar"))
    l2 = get_config("l2")
    if l2:
        cfg.setdefault("l2_redis_host", l2.get("redis_host", "localhost"))
        cfg.setdefault("l2_redis_port", l2.get("redis_port", 6379))
    return _env_overlay("SIDECAR_", cfg)


class SidecarConfig(BaseSettings):
    """Sidecar settings — mirrors ``shared.config.SidecarSection``."""

    model_config = {"env_prefix": "SIDECAR_"}

    host: str = SidecarSection.model_fields["host"].default
    port: int = SidecarSection.model_fields["port"].default
    shared_volume: str = SidecarSection.model_fields["shared_volume"].default
    max_adapters: int = SidecarSection.model_fields["max_adapters"].default
    l1_capacity_mb: int = SidecarSection.model_fields["l1_capacity_mb"].default
    engine_url: str = SidecarSection.model_fields["engine_url"].default
    l2_redis_host: str = SidecarSection.model_fields["l2_redis_host"].default
    l2_redis_port: int = SidecarSection.model_fields["l2_redis_port"].default
    grpc_port: int = SidecarSection.model_fields["grpc_port"].default
    initial_model: str = SidecarSection.model_fields["initial_model"].default
    initial_model_version: str = SidecarSection.model_fields["initial_model_version"].default
    model_store_path: str = SidecarSection.model_fields["model_store_path"].default
    registry_path: str = SidecarSection.model_fields["registry_path"].default
    l1_num_blocks: int = SidecarSection.model_fields["l1_num_blocks"].default
    l1_block_size_bytes: int = SidecarSection.model_fields["l1_block_size_bytes"].default
    # New fields
    hf_token_file: Optional[str] = SidecarSection.model_fields["hf_token_file"].default
    verify_checksums: bool = SidecarSection.model_fields["verify_checksums"].default
    log_json: bool = SidecarSection.model_fields["log_json"].default
    log_level: str = SidecarSection.model_fields["log_level"].default
    otlp_endpoint: Optional[str] = None

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["init_settings"],
            kwargs["env_settings"],
            _sidecar_yaml_source,
        )
