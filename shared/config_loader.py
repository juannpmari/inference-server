"""Centralized YAML config loader.

Reads server_config.yaml once and exposes section-level accessors so that
each service's Pydantic BaseSettings can pull defaults from the YAML while
still allowing environment-variable overrides.

Priority (highest to lowest):
  1. Environment variables
  2. YAML config file
  3. Class-level defaults
"""

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "server_config.yaml"


@lru_cache(maxsize=1)
def _load_yaml(path: str) -> dict[str, Any]:
    """Load and cache the YAML config file."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Config file not found at {p}, using empty config")
        return {}
    with open(p) as f:
        data = yaml.safe_load(f)
    logger.info(f"Loaded server config from {p}")
    return data or {}


def get_config(section: str | None = None) -> dict[str, Any]:
    """Return a config section (or the full dict if *section* is None).

    The path to ``server_config.yaml`` can be overridden with the
    ``SERVER_CONFIG_PATH`` environment variable.
    """
    path = os.environ.get("SERVER_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
    cfg = _load_yaml(path)
    if section is None:
        return cfg
    return cfg.get(section, {})


def yaml_settings_source(section: str):
    """Return a pydantic-settings source callable that loads from YAML.

    Usage in a BaseSettings subclass::

        @classmethod
        def settings_customise_sources(cls, settings_cls, **kwargs):
            return (
                kwargs["env_settings"],         # highest priority
                yaml_settings_source("engine"), # middle priority
                kwargs["init_settings"],        # lowest priority (class defaults)
            )
    """
    def _source() -> dict[str, Any]:
        return get_config(section)
    return _source
