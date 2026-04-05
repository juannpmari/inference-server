"""Distributed tracing helpers using OpenTelemetry."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def init_tracing(service_name: str, otlp_endpoint: Optional[str] = None) -> None:
    """Initialise OpenTelemetry tracing.

    If *otlp_endpoint* is ``None`` the default ``NoOpTracerProvider`` is left
    in place and no spans are exported.
    """
    if otlp_endpoint is None:
        logger.info("Tracing disabled (no OTLP endpoint configured)")
        return

    if not _HAS_OTEL:
        logger.warning("Tracing requested but opentelemetry is not installed")
        return

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("Tracing enabled for service '%s' -> %s", service_name, otlp_endpoint)


def instrument_app(app) -> None:
    """Instrument a FastAPI app and the httpx client library."""
    if not _HAS_OTEL:
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


__all__ = ["init_tracing", "instrument_app"]
