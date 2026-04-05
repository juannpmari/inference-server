"""Gateway Prometheus metrics.

Declares all gateway-level metrics as module-level variables so they are
registered in the default REGISTRY on import.
"""

from prometheus_client import Counter, Histogram, Gauge

# ---------------------------------------------------------------------------
# Core request metrics
# ---------------------------------------------------------------------------

gateway_requests_total = Counter(
    "gateway_requests_total",
    "Total gateway requests",
    ["model", "status_code"],
)

gateway_request_duration_seconds = Histogram(
    "gateway_request_duration_seconds",
    "Gateway request duration",
    ["model"],
)

# ---------------------------------------------------------------------------
# Resilience metrics (populated by WF3 resilience middleware)
# ---------------------------------------------------------------------------

gateway_circuit_breaker_state = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["target"],
)

gateway_rate_limit_rejected_total = Counter(
    "gateway_rate_limit_rejected_total",
    "Requests rejected by rate limiter",
)
