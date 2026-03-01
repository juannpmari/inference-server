# Robustness & Production-Readiness Plan

## Context

After implementing all functional features (Phases A–M in `plan.md`), the system runs on 3 containers — gateway, engine, sidecar — but lacks the hardening needed for production Kubernetes deployment. This plan addresses configuration fragmentation, missing resilience patterns, weak observability, and incomplete K8s integration. The goal: after implementing this plan, the system drops into K8s and serves millions of users with no architectural changes.

**Current gaps**:
- Config scattered across 3 `BaseSettings` classes with no cross-validation (`EngineConfig.model_name` defaults to `"Qwen/Qwen2-0.5B"` while `SidecarConfig.initial_model` defaults to `"Qwen/Qwen2.5-7B-Instruct-1M"` — nothing catches this mismatch)
- Gateway `/health` always returns 200 even when engine is down
- No circuit breakers, no retry-with-backoff (except sidecar download retries in Phase L)
- No request ID propagation, no distributed tracing
- Logging is print-style `%(asctime)s %(levelname)s [%(name)s] %(message)s`, not structured JSON (Phase L adds JSON only for sidecar)
- Registry is a plain JSON file with no locking — concurrent writes corrupt it
- No input validation at gateway (accepts raw `dict`), no rate limiting
- K8s manifests are skeletal shells with empty `command:` fields, no probes, no resource limits
- `MODEL_SERVICE_MAP` in gateway is hardcoded Python, not configurable

---

## Phase R1 — Foundation Layer (Shared Infrastructure)

**Goal**: Build cross-cutting primitives every subsequent phase depends on. No behavioral changes to the three services yet.

### R1.1 Unified Configuration System

| Detail | Value |
|--------|-------|
| **Problem** | Config scattered across 3 files, no cross-validation, hardcoded values |
| **Files to create** | `shared/config.py`, `inference-server.yaml.example` |
| **Files to modify** | `data_plane/gateway/config.py`, `data_plane/inference/engine/config.py`, `data_plane/inference/sidecar/config.py`, `data_plane/gateway/routing.py` |
| **Test** | Mismatched `model_name`/`initial_model` → `ValidationError`; YAML loading + env var overrides work; backward compat with existing env vars |

**Design**: A single `InferenceServerConfig` pydantic model in `shared/config.py` with nested sections (`GatewaySection`, `EngineSection`, `SidecarSection`). A `@model_validator` enforces cross-component rules:

```python
class InferenceServerConfig(BaseModel):
    schema_version: str = "1.0"
    gateway: GatewaySection = GatewaySection()
    engine: EngineSection = EngineSection()
    sidecar: SidecarSection = SidecarSection()

    @model_validator(mode='after')
    def validate_cross_component(self):
        if self.engine.model_name != self.sidecar.initial_model:
            raise ValueError(
                f"engine.model_name={self.engine.model_name!r} != "
                f"sidecar.initial_model={self.sidecar.initial_model!r}"
            )
        if self.gateway.request_timeout <= self.engine.inference_timeout:
            raise ValueError("gateway.request_timeout must exceed engine.inference_timeout")
        return self
```

**Loading strategy**: Check `INFERENCE_SERVER_CONFIG` env var for path to YAML file. If absent, fall back to individual env vars (`GATEWAY_*`, `ENGINE_*`, `SIDECAR_*`) for backward compatibility. Each component imports only its section. The gateway's `MODEL_SERVICE_MAP` becomes `gateway.routes: Dict[str, str]` in config.

**Key fields to extract from hardcoded values**:
- `gateway.routes` (replaces hardcoded `MODEL_SERVICE_MAP` in `routing.py`)
- `engine.inference_timeout` (replaces hardcoded `300.0` in `api.py:195`)
- `engine.drain_timeout` (new, for graceful shutdown)
- `sidecar.hf_token_file` (alternative to plain `HF_TOKEN` env var)
- `sidecar.verify_checksums` (new, default `True`)

---

### R1.2 Structured Error Handling

| Detail | Value |
|--------|-------|
| **Problem** | Inconsistent error formats: gateway uses `{"detail": "..."}`, engine mixes `{"detail":"..."}` and `{"status":"..."}`, no error codes, no request IDs |
| **Files to create** | `shared/errors.py`, `shared/middleware.py` |
| **Files to modify** | `data_plane/gateway/routing.py`, `data_plane/inference/engine/api.py`, `data_plane/inference/sidecar/api.py` |
| **Test** | Error responses match schema; request ID propagates gateway→engine→sidecar; unknown error codes don't crash |

**Design**: `shared/errors.py` defines `ErrorCode` enum (1xxx=gateway, 2xxx=engine, 3xxx=sidecar) and `ErrorResponse` model:

```python
class ErrorCode(IntEnum):
    MODEL_NOT_FOUND = 1001
    ENGINE_UNREACHABLE = 1002
    RATE_LIMITED = 1004
    ENGINE_NOT_READY = 2001
    QUEUE_FULL = 2002
    INFERENCE_TIMEOUT = 2003
    SIDECAR_NOT_READY = 3001
    MODEL_DOWNLOAD_FAILED = 3002
    DISK_SPACE_INSUFFICIENT = 3005

class ErrorResponse(BaseModel):
    error_code: int
    message: str
    request_id: Optional[str] = None
    details: Optional[dict] = None
```

`shared/middleware.py` provides `RequestIDMiddleware` — generates UUID at gateway (or accepts `X-Request-ID` header), propagates via header to downstream services, injects into all log records and error responses.

---

### R1.3 Structured Logging

| Detail | Value |
|--------|-------|
| **Problem** | Engine uses `logging.basicConfig(format="%(asctime)s %(levelname)s...")` — not JSON, no request IDs, different formats per component |
| **Files to create** | `shared/logging_config.py` |
| **Files to modify** | All three `api.py` files (replace `logging.basicConfig` with `configure_logging("component")`) |
| **Test** | Capture log output → parse as JSON → verify structure; request_id appears when set |

**Design**: `shared/logging_config.py` provides `JSONFormatter` and `configure_logging(component)`. Uses `contextvars` to inject `request_id` into all log records within a request scope (works with `RequestIDMiddleware` from R1.2).

```python
def configure_logging(component: str, level: str = "INFO", json_output: bool = True):
    """Call once at module level in each service."""
```

---

### R1 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R1 Verification ==="

# 1. Cross-component config validation
echo "[1/4] Testing config cross-validation..."
uv run python -c "
from shared.config import InferenceServerConfig
try:
    InferenceServerConfig(
        engine={'model_name': 'model-A'},
        sidecar={'initial_model': 'model-B'}
    )
    assert False, 'Should have raised'
except ValueError as e:
    assert 'model_name' in str(e)
    print('  Cross-validation catches mismatch')
"

# 2. YAML config loads correctly
echo "[2/4] Testing YAML loading..."
uv run python -c "
from shared.config import load_config
config = load_config('inference-server.yaml.example')
assert config.schema_version
print(f'  Loaded config v{config.schema_version}')
"

# 3. Error responses match schema
echo "[3/4] Testing error response schema..."
uv run pytest tests/unit/test_errors.py -v --tb=short

# 4. JSON logging produces valid output
echo "[4/4] Testing structured logging..."
uv run pytest tests/unit/test_logging.py -v --tb=short

echo "=== Phase R1: ALL CHECKS PASSED ==="
```

---

## Phase R2 — Resilience Patterns

**Depends on**: Phase R1 (uses error codes, request IDs, structured logging)

### R2.1 Circuit Breaker & Retry with Backoff

| Detail | Value |
|--------|-------|
| **Problem** | Engine hammers sidecar every 2s for 600s when sidecar is down. Gateway tries every request even when engine is unreachable. No retry on transient failures. |
| **Files to create** | `shared/resilience.py` |
| **Files to modify** | `data_plane/gateway/routing.py` (circuit breaker around engine calls), `data_plane/inference/engine/api.py` (backoff on sidecar polling, circuit breaker for runtime calls), `data_plane/inference/engine/engine.py` (retry on adapter fetch) |
| **Test** | Circuit breaker: closed→open→half-open→closed transitions; after N failures, calls fail-fast without network attempt; backoff timing with mocked `asyncio.sleep` |

**Design**: `shared/resilience.py` implements:

1. **`CircuitBreaker`** — three states (CLOSED/OPEN/HALF_OPEN), configurable `failure_threshold` (default 5), `recovery_timeout` (default 30s), `half_open_max_calls` (default 1)
2. **`retry_with_backoff()`** — `max_retries=3`, `base_delay=1.0`, exponential up to `max_delay=30.0`, configurable `retryable_exceptions`

**Application points**:
- Gateway→Engine: wrap `http_client.post()` in circuit breaker. When open, return 503 immediately.
- Engine→Sidecar polling: replace flat 2s interval with backoff (2s, 4s, 8s, capped at 30s)
- Engine→Sidecar adapter fetch: wrap in `retry_with_backoff` for transient failures

---

### R2.2 Health Check Cascading

| Detail | Value |
|--------|-------|
| **Problem** | Gateway `/health` always returns 200 — K8s thinks gateway is healthy even when engine is down |
| **Files to modify** | `data_plane/gateway/routing.py` (add `/ready` endpoint that checks engine), K8s manifests (define readinessProbe pointing to `/ready`) |
| **Test** | Mock engine returning 503 → gateway `/ready` returns 503; cached result TTL works (5s) |

**Design**: Gateway gets `/ready` endpoint that calls `engine:8080/ready` with 2s timeout. Result cached for 5s to avoid hammering. Returns 503 with `ErrorCode.ENGINE_UNREACHABLE` when engine is not ready.

---

### R2.3 Timeout Cascading

| Detail | Value |
|--------|-------|
| **Problem** | Gateway and engine both use 300s timeout — ambiguous when they expire simultaneously |
| **Files to modify** | `shared/config.py` (add validator), `engine/api.py` (use config value), `gateway/routing.py` (use config value) |
| **Test** | Config validator rejects `gateway.request_timeout <= engine.inference_timeout + 10` |

**Rule**: `gateway.request_timeout (300s) > engine.inference_timeout (270s) + 10s buffer`. Cross-component validator enforces this.

---

### R2.4 Rate Limiting at Gateway

| Detail | Value |
|--------|-------|
| **Problem** | No rate limiting — single client can flood the system |
| **Files to create** | `shared/rate_limiter.py` |
| **Files to modify** | `data_plane/gateway/routing.py` (add rate limiting middleware), config (add `rate_limit_rps`, `rate_limit_burst`) |
| **Test** | Send burst above limit → 429 with `Retry-After`; within limit → 200 |

**Design**: In-memory token-bucket rate limiter. Returns 429 with `Retry-After` header. Config fields: `gateway.rate_limit_rps: float = 100.0`, `gateway.rate_limit_burst: int = 200`. Note: for multi-instance gateway, swap to Redis-backed limiter (out of scope for this plan — K8s HPA handles it via per-pod limiting).

---

### R2 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R2 Verification ==="

# 1. Circuit breaker unit tests
echo "[1/3] Running circuit breaker tests..."
uv run pytest tests/unit/test_resilience.py -v --tb=short

# 2. Health cascading
echo "[2/3] Testing health cascading..."
uv run pytest tests/unit/test_gateway_health.py -v --tb=short

# 3. Rate limiter
echo "[3/3] Running rate limiter tests..."
uv run pytest tests/unit/test_rate_limiter.py -v --tb=short

echo "=== Phase R2: ALL CHECKS PASSED ==="
```

---

## Phase R3 — Data Integrity & Security

**Independent of Phase R2** — can run in parallel.

### R3.1 Registry File Locking & Atomic Writes

| Detail | Value |
|--------|-------|
| **Problem** | `artifact_manager.py:_persist_registry()` does plain `open("w") + json.dump()` — concurrent writes corrupt the file |
| **Files to modify** | `data_plane/inference/sidecar/artifact_manager.py` |
| **Test** | Concurrent `_persist_registry` calls → valid JSON; simulated crash mid-write → old file intact |

**Design**: Two-layer protection:
1. **Application level**: `asyncio.Lock` in `ArtifactManager` serializes all registry mutations
2. **File level**: Write to temp file (`tempfile.mkstemp`), then `os.replace()` for atomic rename. Use `fcntl.flock(LOCK_EX)` for OS-level file locking as defense-in-depth.

---

### R3.2 Disk Space Pre-checks

| Detail | Value |
|--------|-------|
| **Problem** | `_fetch_from_external_storage` downloads blindly — disk-full mid-download causes partial files and confusing `OSError` |
| **Files to modify** | `data_plane/inference/sidecar/artifact_manager.py` |
| **Test** | Mock `shutil.disk_usage` with low free space → `DiskSpaceError` before download starts |

**Design**: Call `shutil.disk_usage(shared_volume)` before download. Require at least `max(10GB, 10% of total)`. Raise `DiskSpaceError` (with `ErrorCode.DISK_SPACE_INSUFFICIENT`) if insufficient.

---

### R3.3 Model Checksum Verification

| Detail | Value |
|--------|-------|
| **Problem** | Downloaded models are not verified after download — corruption goes undetected |
| **Files to modify** | `data_plane/inference/sidecar/artifact_manager.py` |
| **Test** | Download model → checksum stored in registry; load from cache → checksum matches; corrupt file → verification fails |

**Design**: After download, compute SHA256 of key files (config.json, safetensors), store in registry entry. On subsequent loads from cache, verify checksum. Configurable via `sidecar.verify_checksums: bool = True`. Note: `huggingface_hub.snapshot_download` already verifies via etags — this is defense-in-depth.

---

### R3.4 Secret Management

| Detail | Value |
|--------|-------|
| **Problem** | `HF_TOKEN` is a plain env var — not suitable for K8s production |
| **Files to modify** | `shared/config.py` (add `hf_token_file` field), `artifact_manager.py` (read token from file), K8s manifests (add Secret volume mount) |
| **Test** | Write token to file → sidecar reads it; no file + no env var → clear error |

**Design**: Add `sidecar.hf_token_file: Optional[str]` to config. If set, read token from file path (K8s Secret mounted as file). Falls back to `HF_TOKEN` env var. K8s manifest mounts `secretKeyRef` to `/etc/secrets/hf-token`.

---

### R3.5 Input Validation & Request Size Limits

| Detail | Value |
|--------|-------|
| **Problem** | Gateway `/generate` accepts raw `dict` (line 41 of `routing.py`: `event: dict`) — no validation, no size limits |
| **Files to modify** | `data_plane/gateway/routing.py` |
| **Test** | Send 2MB body → 413; send invalid model name → 400; send valid request → 200 |

**Design**: Replace `event: dict` with proper `GenerateRequest` pydantic model:
```python
class GenerateRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
```
Add middleware that rejects request bodies > 1MB (HTTP 413).

---

### R3 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R3 Verification ==="

echo "[1/4] Registry locking tests..."
uv run pytest tests/unit/ -k "registry_lock or atomic_write" -v --tb=short

echo "[2/4] Disk space pre-check tests..."
uv run pytest tests/unit/ -k "disk_space" -v --tb=short

echo "[3/4] Input validation tests..."
uv run pytest tests/unit/ -k "input_validation or request_size" -v --tb=short

echo "[4/4] Secret management tests..."
uv run pytest tests/unit/ -k "secret or hf_token" -v --tb=short

echo "=== Phase R3: ALL CHECKS PASSED ==="
```

---

## Phase R4 — Observability

**Depends on**: Phase R1 (structured logging, request IDs)

### R4.1 OpenTelemetry Distributed Tracing

| Detail | Value |
|--------|-------|
| **Problem** | No distributed tracing — impossible to debug latency across gateway→engine→sidecar |
| **Files to create** | `shared/tracing.py` |
| **Files to modify** | `pyproject.toml` (add OTel deps), all three `api.py` files (init tracing in lifespan), `docker-compose.yml` (add Jaeger service) |
| **Test** | Send request → Jaeger shows trace spanning all 3 services |

**Design**: `shared/tracing.py` wraps OpenTelemetry SDK setup:
- `init_tracing(service_name, otlp_endpoint)` — creates `TracerProvider` with OTLP exporter
- `instrument_app(app)` — instruments FastAPI + httpx
- Tracing disabled when `otlp_endpoint` is `None` (default) — opt-in via config

Dependencies to add:
```
opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp,
opentelemetry-instrumentation-fastapi, opentelemetry-instrumentation-httpx
```

---

### R4.2 Enhanced Prometheus Metrics & Alerting

| Detail | Value |
|--------|-------|
| **Problem** | Gateway has zero metrics (no `/metrics` endpoint). No alerting rules. |
| **Files to create** | `data_plane/gateway/metrics.py`, `docker/alerting-rules.yml` |
| **Files to modify** | `data_plane/gateway/routing.py` (add `/metrics`, instrument routes), `docker/prometheus.yml` |
| **Test** | Hit gateway `/metrics` → valid Prometheus text format; alerting rules parse without error |

**Gateway metrics to add**:
- `gateway_requests_total` (Counter, labels: model, status_code)
- `gateway_request_duration_seconds` (Histogram, labels: model)
- `gateway_circuit_breaker_state` (Gauge, labels: target)
- `gateway_rate_limit_rejected_total` (Counter)

**Alerting rules**:
- `HighErrorRate`: 5xx rate > 5% for 2min → critical
- `HighLatency`: p99 > 30s for 5min → warning
- `QueueSaturation`: pending/max > 80% for 1min → warning
- `CircuitBreakerOpen`: any circuit open for 30s → critical

---

### R4 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R4 Verification ==="

echo "[1/3] Gateway metrics endpoint..."
uv run python -c "
from data_plane.gateway import metrics
attrs = [a for a in dir(metrics) if 'gateway' in a.lower()]
assert len(attrs) >= 3, f'Expected gateway metrics, got: {attrs}'
print(f'  Gateway metrics: {attrs}')
"

echo "[2/3] Tracing module importable..."
uv run python -c "
from shared.tracing import init_tracing, instrument_app
print('  Tracing module OK')
"

echo "[3/3] Alerting rules valid YAML..."
uv run python -c "
import yaml
with open('docker/alerting-rules.yml') as f:
    rules = yaml.safe_load(f)
assert 'groups' in rules
print(f'  {len(rules[\"groups\"][0][\"rules\"])} alerting rules defined')
"

echo "=== Phase R4: ALL CHECKS PASSED ==="
```

---

## Phase R5 — Startup Orchestration & Lifecycle

**Depends on**: Phase R1, R2

### R5.1 Pre-flight Checks

| Detail | Value |
|--------|-------|
| **Problem** | Services start and fail late with confusing errors (e.g., no GPU, disk full, sidecar unreachable) |
| **Files to create** | `shared/preflight.py` |
| **Files to modify** | All three `api.py` files (call `run_preflight()` in lifespan) |
| **Test** | Mock GPU unavailable → clear error at startup; mock disk full → clear error |

**Pre-flight checks per component**:

| Component | Check | Critical? |
|-----------|-------|-----------|
| Engine | GPU available (unless mock mode) | Yes |
| Engine | Sidecar URL is valid URL format | Yes |
| Engine | Sufficient system memory | Warning |
| Sidecar | Shared volume writable | Yes |
| Sidecar | Disk space > 10GB | Yes |
| Sidecar | Redis reachable (if L2 configured) | Warning |
| Sidecar | HF_TOKEN set (if private models) | Warning |
| Gateway | At least one route configured | Yes |
| Gateway | Engine URL is valid URL format | Yes |

---

### R5.2 K8s Probe Separation (healthz / readyz / startupz)

| Detail | Value |
|--------|-------|
| **Problem** | Engine `/health` returns 503 during model loading — K8s liveness probe would kill the pod before model loads |
| **Files to modify** | All three `api.py` files, all K8s manifests |
| **Test** | During model loading: `/healthz`→200, `/readyz`→503, `/startupz`→503; after load: all→200 |

**Design**: Each service gets three endpoints:
- **`/healthz`** (liveness) — is the process alive? Always 200 after app starts. K8s uses this to decide whether to restart.
- **`/readyz`** (readiness) — can this service handle traffic? Checks dependencies. K8s uses this to include/exclude from Service endpoints.
- **`/startupz`** (startup) — has initial setup completed? K8s uses this to delay liveness probes. For engine: 200 only after model loaded (`failureThreshold: 120, periodSeconds: 5` = 10min startup budget).

Keep `/health` and `/ready` as aliases for backward compat.

---

### R5.3 Graceful Shutdown with Request Draining

| Detail | Value |
|--------|-------|
| **Problem** | Engine shutdown cancels init task and batching loop — active requests get `CancelledError` |
| **Files to modify** | `data_plane/inference/engine/api.py`, `data_plane/gateway/routing.py` |
| **Test** | Submit request → send SIGTERM → request completes successfully → new requests get 503 |

**Design**:
1. On SIGTERM, set `_accepting_requests = False`
2. `/inference` returns 503 with `Retry-After: 1` when not accepting (K8s routes to another pod)
3. Wait for in-flight requests to complete (configurable `drain_timeout`, default 30s)
4. After drain timeout, cancel remaining requests
5. Cancel batching loop
6. K8s `terminationGracePeriodSeconds: 60` > `drain_timeout: 30` (buffer for cleanup)

---

### R5.4 Model Download Progress Reporting

| Detail | Value |
|--------|-------|
| **Problem** | Engine polls sidecar `/registry/models` and only sees `status: "downloading"` — no progress info |
| **Files to modify** | `sidecar/artifact_manager.py`, `sidecar/api.py` |
| **Test** | Start model download → GET `/status/{model}` → returns `downloaded_bytes`, `status` |

**Design**: Add `download_progress: Dict[str, dict]` to `ArtifactManager`. Track progress by periodically checking temp directory size. New endpoint: `GET /status/{model_identifier:path}` returns registry entry + progress info.

---

### R5 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R5 Verification ==="

echo "[1/3] Pre-flight checks..."
uv run pytest tests/unit/ -k "preflight" -v --tb=short

echo "[2/3] Probe separation..."
uv run python -c "
import inspect
from data_plane.inference.engine import api as engine_api
source = inspect.getsource(engine_api)
for probe in ['healthz', 'readyz', 'startupz']:
    assert probe in source, f'Missing /{probe} endpoint'
print('  All probe endpoints present')
"

echo "[3/3] Graceful shutdown drain logic..."
uv run pytest tests/unit/ -k "shutdown or drain" -v --tb=short

echo "=== Phase R5: ALL CHECKS PASSED ==="
```

---

## Phase R6 — Kubernetes Production Manifests

**Depends on**: Phase R5 (probe endpoints, config system)

### R6.1 Complete K8s Manifests

| Detail | Value |
|--------|-------|
| **Problem** | Manifests are skeletal shells — empty `command:` fields, no probes, no resource limits |
| **Files to create** | `manifests/06-configmap.yaml`, `manifests/07-secret.yaml`, `manifests/08-pvc.yaml`, `manifests/09-hpa.yaml` |
| **Files to modify** | `manifests/04-gateway-deploy.yaml`, `manifests/05-inference-service-deploy.yaml` |
| **Test** | `kubectl apply --dry-run=client -f manifests/` succeeds; manifests reference correct probe paths |

**Key manifest content**:

**Inference pod (engine + sidecar in same pod)**:
```yaml
spec:
  terminationGracePeriodSeconds: 60
  containers:
  - name: engine
    resources:
      limits: { nvidia.com/gpu: 1, memory: 32Gi }
      requests: { memory: 16Gi }
    startupProbe:   { httpGet: {path: /startupz, port: 8080}, failureThreshold: 120, periodSeconds: 5 }
    livenessProbe:  { httpGet: {path: /healthz,  port: 8080}, periodSeconds: 10, failureThreshold: 3 }
    readinessProbe: { httpGet: {path: /readyz,   port: 8080}, periodSeconds: 5 }
  - name: sidecar
    startupProbe:   { httpGet: {path: /startupz, port: 8001}, failureThreshold: 120, periodSeconds: 5 }
    livenessProbe:  { httpGet: {path: /healthz,  port: 8001}, periodSeconds: 10 }
    readinessProbe: { httpGet: {path: /readyz,   port: 8001}, periodSeconds: 5 }
```

**New manifests**:
- `06-configmap.yaml` — mounts `inference-server.yaml` from ConfigMap
- `07-secret.yaml` — template for HF_TOKEN Secret
- `08-pvc.yaml` — PersistentVolumeClaim for `/mnt/models` (ReadWriteMany for shared access)
- `09-hpa.yaml` — HorizontalPodAutoscaler for gateway (scale on CPU/request rate)

---

### R6 — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase R6 Verification ==="

echo "[1/2] Manifests are valid YAML..."
for f in manifests/*.yaml; do
  uv run python -c "import yaml; yaml.safe_load(open('$f'))" || { echo "FAIL: $f"; exit 1; }
done
echo "  All manifests valid"

echo "[2/2] Probe paths match code..."
uv run python -c "
import yaml
with open('manifests/05-inference-service-deploy.yaml') as f:
    spec = yaml.safe_load(f)
containers = spec['spec']['template']['spec']['containers']
for c in containers:
    for probe_type in ['startupProbe', 'livenessProbe', 'readinessProbe']:
        if probe_type in c:
            path = c[probe_type]['httpGet']['path']
            assert path.startswith('/'), f'{c[\"name\"]} {probe_type} path invalid: {path}'
            print(f'  {c[\"name\"]}.{probe_type} -> {path}')
"

echo "=== Phase R6: ALL CHECKS PASSED ==="
```

---

## Phase R7 — Testing Strategy

**Incrementally throughout**, with integration/chaos/load tests after Phase R5.

### R7.1 Unit Tests for Shared Modules

| Test File | Covers |
|-----------|--------|
| `tests/unit/test_config.py` | Unified config validation, cross-component checks, YAML loading, env var override, backward compat |
| `tests/unit/test_errors.py` | Error response construction, error code enum, request ID injection |
| `tests/unit/test_resilience.py` | Circuit breaker state machine (closed→open→half-open→closed), retry backoff timing |
| `tests/unit/test_rate_limiter.py` | Token bucket under burst, steady state, refill behavior |
| `tests/unit/test_preflight.py` | Pre-flight check framework, critical vs warning checks |

### R7.2 Integration Tests (docker-compose)

| Test File | Scenario |
|-----------|----------|
| `tests/integration/test_e2e_health.py` | Kill sidecar → engine unhealthy → gateway unhealthy |
| `tests/integration/test_e2e_inference.py` | Full inference flow with mock engine |
| `tests/integration/test_e2e_circuit_breaker.py` | Kill engine → gateway circuit opens → restart → recovery |
| `tests/integration/test_e2e_graceful_shutdown.py` | SIGTERM during inference → request completes |

Create `tests/integration/docker-compose.test.yml` with mock engine enabled.

### R7.3 Load Testing

| File | Purpose |
|------|---------|
| `tests/load/locustfile.py` | Simulate concurrent users hitting `/generate` |

### R7.4 Chaos Testing

| Script | Scenario |
|--------|----------|
| `tests/chaos/kill_random.sh` | Kill random containers, verify recovery |
| `tests/chaos/network_delay.sh` | Introduce latency via `tc`, verify timeouts work |
| `tests/chaos/disk_full.sh` | Fill disk, verify pre-check catches it |

---

## Implementation Sequencing

```
Phase R1 (Foundation) — do first, no dependencies
  R1.1 Unified Config ─────┐
  R1.2 Structured Errors ──┼── all independent, can parallelize
  R1.3 Structured Logging ─┘

Phase R2 (Resilience) — depends on R1
  R2.1 Circuit Breaker + Retry  (uses errors, logging)
  R2.2 Health Cascading         (uses errors, config)
  R2.3 Timeout Cascading        (uses config)
  R2.4 Rate Limiting            (uses errors, config)

Phase R3 (Data/Security) — independent of R2, can run in parallel
  R3.1 Registry Locking    (independent)
  R3.2 Disk Pre-checks     (independent)
  R3.3 Checksums           (independent)
  R3.4 Secret Management   (independent)
  R3.5 Input Validation    (uses errors from R1.2)

Phase R4 (Observability) — depends on R1
  R4.1 OpenTelemetry       (uses logging, request IDs)
  R4.2 Enhanced Metrics    (uses circuit breaker from R2.1)

Phase R5 (Lifecycle) — depends on R1, R2
  R5.1 Pre-flight Checks   (uses config, logging)
  R5.2 Probe Separation    (uses health cascading from R2.2)
  R5.3 Graceful Shutdown   (uses logging)
  R5.4 Progress Reporting  (independent)

Phase R6 (K8s Manifests) — depends on R5
  R6.1 Complete Manifests  (uses probes from R5.2, config from R1.1)

Phase R7 (Testing) — incremental
  R7.1 Unit tests          (per phase)
  R7.2-R7.4 Integration/Load/Chaos (after R5)
```

---

## File Summary

### New Files (18)

| File | Phase | Purpose |
|------|-------|---------|
| `shared/config.py` | R1.1 | Unified config with cross-validation |
| `shared/errors.py` | R1.2 | Error codes, structured error responses |
| `shared/middleware.py` | R1.2 | Request ID middleware |
| `shared/logging_config.py` | R1.3 | JSON structured logging |
| `shared/resilience.py` | R2.1 | Circuit breaker, retry with backoff |
| `shared/rate_limiter.py` | R2.4 | Token bucket rate limiter |
| `shared/preflight.py` | R5.1 | Pre-flight check framework |
| `shared/tracing.py` | R4.1 | OpenTelemetry initialization |
| `data_plane/gateway/metrics.py` | R4.2 | Gateway Prometheus metrics |
| `docker/alerting-rules.yml` | R4.2 | Prometheus alerting rules |
| `inference-server.yaml.example` | R1.1 | Example unified config file |
| `manifests/06-configmap.yaml` | R6.1 | K8s ConfigMap |
| `manifests/07-secret.yaml` | R6.1 | K8s Secret template |
| `manifests/08-pvc.yaml` | R6.1 | PersistentVolumeClaim |
| `manifests/09-hpa.yaml` | R6.1 | HorizontalPodAutoscaler |
| `tests/unit/test_config.py` | R7.1 | Config validation tests |
| `tests/unit/test_resilience.py` | R7.1 | Circuit breaker + retry tests |
| `tests/unit/test_rate_limiter.py` | R7.1 | Rate limiter tests |

### Modified Files (14)

| File | Phases | Key Changes |
|------|--------|-------------|
| `data_plane/gateway/routing.py` | R1–R5 | Config-driven routes, `/ready`, `/metrics`, circuit breaker, rate limiter, request validation, request ID, structured errors/logging, tracing, graceful shutdown, probe separation |
| `data_plane/gateway/config.py` | R1.1 | Delegate to `shared.config` |
| `data_plane/inference/engine/api.py` | R1–R5 | Structured logging/errors, request ID, configurable timeouts, circuit breaker on sidecar calls, backoff on polling, graceful drain, tracing, probe separation |
| `data_plane/inference/engine/config.py` | R1.1 | Delegate to `shared.config` |
| `data_plane/inference/engine/engine.py` | R2.1 | Retry on adapter fetch |
| `data_plane/inference/sidecar/api.py` | R1–R5 | Structured logging/errors, request ID, tracing, probe separation, `/status` endpoint |
| `data_plane/inference/sidecar/config.py` | R1.1 | Delegate to `shared.config` |
| `data_plane/inference/sidecar/artifact_manager.py` | R3 | File locking, atomic writes, disk pre-checks, checksums, asyncio.Lock, progress tracking, token from file |
| `pyproject.toml` | R4.1 | Add OpenTelemetry + locust dependencies |
| `docker-compose.yml` | R4.1 | Add Jaeger service |
| `docker/prometheus.yml` | R4.2 | Add alerting rules reference |
| `manifests/04-gateway-deploy.yaml` | R6.1 | Full rewrite with probes, resources |
| `manifests/05-inference-service-deploy.yaml` | R6.1 | Full rewrite with probes, resources, secrets |
| `.env.example` | R1.1 | Add `INFERENCE_SERVER_CONFIG` reference |
