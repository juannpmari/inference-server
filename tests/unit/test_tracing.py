"""Unit tests for shared.tracing and OTel config fields."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.util._once import Once

from shared.tracing import init_tracing, instrument_app


def _force_reset_tracer_provider():
    """Force-reset the global TracerProvider so tests can call set_tracer_provider again."""
    # Reset the once-guard so set_tracer_provider will accept a new value.
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Ensure each test starts and ends with a clean NoOp state."""
    _force_reset_tracer_provider()
    yield
    _force_reset_tracer_provider()


def test_init_tracing_noop_when_no_endpoint():
    init_tracing("test-svc", None)
    span = trace.get_tracer("test").start_span("x")
    assert isinstance(span, NonRecordingSpan)


def test_init_tracing_sets_real_provider():
    init_tracing("test-svc", "http://localhost:4317")
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    # Verify service.name resource attribute
    resource_attrs = provider.resource.attributes
    assert resource_attrs.get("service.name") == "test-svc"


def test_instrument_app_smoke():
    from fastapi import FastAPI

    app = FastAPI()
    instrument_app(app)  # should not raise


def test_config_otlp_endpoint_defaults_none():
    from data_plane.gateway.config import GatewayConfig
    from data_plane.inference.engine.config import EngineConfig
    from data_plane.inference.sidecar.config import SidecarConfig

    gw = GatewayConfig()
    assert gw.otlp_endpoint is None

    eng = EngineConfig()
    assert eng.otlp_endpoint is None

    sc = SidecarConfig()
    assert sc.otlp_endpoint is None
