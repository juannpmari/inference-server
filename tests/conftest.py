import pytest

from data_plane.gateway.config import GatewayConfig
from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.sidecar.config import SidecarConfig


@pytest.fixture
def engine_config():
    return EngineConfig()


@pytest.fixture
def sidecar_config():
    return SidecarConfig()


@pytest.fixture
def gateway_config():
    return GatewayConfig()
