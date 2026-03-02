# Requirements and Architectural Decisions

## 1. Functional Requirements

| ID | Requirement | Mapped Step |
|----|------------|-------------|
| FR-1 | Accept HTTP POST with `{model_id, prompt, max_tokens, temperature, stream}`, return text or SSE stream | 1, 2 |
| FR-2 | Load/unload/hot-swap LLM models at runtime via sidecar without restarting engine | 4 |
| FR-3 | Apply/unapply LoRA adapters on a resident base model; multiple adapters share base weights | 5 |
| FR-4 | L1 (CPU DRAM) and L2 (Redis) KV cache with automatic tiered fallback | 7 |
| FR-5 | Gateway routes requests to pod maximizing cache reuse (prefix hash affinity) | 8 |
| FR-6 | Model/adapter registry with persistence across restarts | 6 |
| FR-7 | SSE streaming of tokens as generated, with client disconnect cancellation | 2 |
| FR-8 | `/health` and `/ready` probes on every service | 1, 4 |
| FR-9 | Queue limits (429), request timeouts (504), graceful shutdown | 9 |
| FR-10 | Prometheus `/metrics` endpoint on every service | 1, 4, 8 |

---

## 2. Non-Functional Requirements

| NFR | Value | Notes |
|-----|-------|-------|
| **Target dev model** | `Qwen/Qwen2-0.5B` | ~1GB VRAM, fast iteration |
| **GPU** | NVIDIA 8GB (RTX 3060/4060) | 8GB VRAM budget |
| **Max concurrent requests** | 10 | Starting point |
| **Latency SLO** | TTFT < 500ms, generation < 50ms/token | Track via metrics, enforce later |
| **Max context length** | 512 (dev) | Keep low for fast tests |
| **GPU memory utilization** | 0.70 | Leave ~2.4GB for OS/desktop |
| **L1 cache budget** | 512 MB | CPU DRAM for KV cache |
| **L2 cache budget** | 256 MB | Redis maxmemory |
| **Adapter limit** | 10 | `MAX_RESIDENT_ADAPTERS` |
| **Model artifact source** | HuggingFace Hub | `huggingface_hub.snapshot_download()` |

### Development Defaults (environment variables)

```
ENGINE_MODEL_NAME        = "Qwen/Qwen2-0.5B"
ENGINE_MAX_MODEL_LEN     = 512
ENGINE_GPU_MEM_UTIL      = 0.70
ENGINE_DTYPE             = "bfloat16"
SIDECAR_L1_CAPACITY_MB   = 512
L2_REDIS_MAXMEM          = 256mb
ENGINE_MAX_PENDING        = 10
HF_TOKEN                 = <from .env file>
ENABLE_ENGINE_MOCK       = false
```

**Mock Engine Note**: When `ENABLE_ENGINE_MOCK=true`, the engine uses deterministic mock responses instead of vLLM. All metrics, routing, batching, streaming, and error handling remain fully functional, allowing complete system testing on machines without GPUs.

---

## 3. Service Level Objectives (SLOs) and Service Level Agreements (SLAs)

### Latency SLOs

| Indicator | Objective | Measured By | Enforcement |
|-----------|-----------|-------------|-------------|
| **Time to First Token (TTFT)** | < 500ms (p95) | `engine_time_to_first_token_seconds` histogram | Track via metrics; enforce later |
| **Token generation latency** | < 50ms/token (p95) | Derived from `engine_tokens_per_second` gauge | Track via metrics; enforce later |
| **Gateway request duration** | < 2s end-to-end (p95) | `gateway_request_duration_seconds` histogram | Track via metrics |
| **Model load time** | < 30s (Qwen2-0.5B) | `sidecar_model_load_duration_seconds` histogram | Informational |
| **LoRA adapter load** | < 5s (download + GPU load) | `engine_lora_load_duration_seconds` histogram | Timeout at `adapter_poll_timeout` (600s) |

### Throughput SLOs

| Indicator | Objective | Measured By |
|-----------|-----------|-------------|
| **Concurrent requests** | 10 max | `engine_pending_requests` gauge |
| **Queue overflow** | Return 429 when pending > 10 | `engine_requests_total{status="429"}` counter |

### Resource Budget SLAs (hard limits)

| Resource | Budget | Alert Threshold | Metric |
|----------|--------|-----------------|--------|
| **GPU VRAM** | 8 GB total, 0.70 utilization (5.6 GB usable) | 90% (7.2 GB) | `engine_gpu_memory_used_bytes` |
| **L1 cache (CPU DRAM)** | 512 MB | — | `cache_l1_used_bytes` / `cache_l1_capacity_bytes` |
| **L2 cache (Redis)** | 256 MB (`maxmemory`) | — | `cache_l2_active_nodes` |
| **LoRA adapter slots** | 4 concurrent on GPU (`max_loras`) | — | `engine_lora_active` |
| **Resident adapters** | 10 (`MAX_RESIDENT_ADAPTERS`) | — | `sidecar_resident_adapters` |

### Timeout and Error SLAs

| Scenario | Behavior | HTTP Status |
|----------|----------|-------------|
| Request exceeds queue limit | Rejected immediately | 429 Too Many Requests |
| Request exceeds timeout | Cancelled, resources freed | 504 Gateway Timeout |
| Client disconnects mid-stream | Generation aborted, queue cleaned up | — |
| Adapter poll exceeds `adapter_poll_timeout` (600s) | Request fails | 504 Gateway Timeout |
| Invalid request payload | Rejected by Pydantic validation | 422 Unprocessable Entity |
| Engine internal error | Structured error response | 500 Internal Server Error |
| Graceful shutdown | In-flight requests drain, new requests rejected | 503 Service Unavailable |

### Availability (target, not yet enforced)

| Indicator | Objective | Notes |
|-----------|-----------|-------|
| **Service uptime** | 99.5% (single-node dev) | Measured via `/health` and `/ready` probes |
| **Health probe response** | < 100ms | `/health` on every service |
| **Readiness probe** | Reflects actual model state | `/ready` returns 503 until model loaded |

> **Note**: SLOs are tracked via Prometheus metrics from day one (Steps 1, 4, 8). Enforcement (auto-scaling, circuit breaking) is deferred to post-Step 10.

---

## 4. Architectural Decision Record: Modular + Selective Ports: Modular + Selective Ports

### Problem

The codebase is a Kubernetes-native LLM inference platform (~28% complete, ~1,300 lines across 15 files). Architecture must be chosen to take it from broken scaffolding to a locally testable system. Solo developer, no business domain entities — this is infrastructure code.

### Options Evaluated

| Architecture | Verdict | Reason |
|-------------|---------|--------|
| **Full Hexagonal (Ports & Adapters)** | Rejected | ~1-2 weeks of restructuring for 1,300 lines of prototype code. Would triple file count with ~20 interface files. Solo developer, no business domain entities. |
| **Clean Architecture (Onion)** | Rejected | Entities (`Model`, `CacheEntry`) have no behavior — they are plain data. Use-case classes would wrap 15-line methods with zero added value. This is infrastructure, not a business domain. |
| **Simple Layered (API->Service->Repository)** | Partially adopted | Good fit for gateway and engine. Redundant for sidecar which already has natural layers. |
| **Modular + Selective Ports** | **Adopted** | 80% of hexagonal's testability benefit at 15% of the cost. Targeted improvements where they matter. |

### Decision: Pragmatic Modular Architecture

1. **Keep existing service boundaries** — gateway, engine, sidecar, controller map to deployment units.
2. **Add `shared/` package** — single definition for shared types (see section 5 below), eliminating cross-file duplication.
3. **Extract configuration** — one `config.py` per service using `pydantic-settings`, replacing 10+ hardcoded constants.
4. **Add 3 Protocol interfaces** — only where testability demands it (see section 4 below).
5. **No use-case classes, no entity classes, no DI container** — existing methods ARE the use cases.

### What Already Works (don't change)

- `EvictionPolicy` base class + `LRUPolicy` in `eviction_policy.py` — already the port/adapter pattern.
- `L1CacheAPI` as facade coordinating allocator + transfer + eviction — correct layering.
- Sidecar pattern (2 containers/pod), control/data plane separation, consistent hashing — all correct.

### When to Revisit

Move toward full hexagonal IF: team grows beyond 2-3 devs, a second cache backend is added (e.g. InfiniStore), sidecar exceeds ~2,000 lines, or multiple deployment contexts are needed.

---

## 4. Protocol Interfaces (3 Selective Ports)

Three `typing.Protocol` interfaces are introduced in `shared/ports.py`, targeting only the seams where testability demands abstraction:

### 4.1 `CacheStore`

- **Methods**: `put(key, data)`, `get(key)`
- **Replaces**: Direct coupling to concrete L1 and L2 cache implementations
- **Why**: Both L1 (CPU DRAM) and L2 (Redis) already conform to this interface. Formalizing it enables injecting `FakeCacheStore` in unit tests without Redis or memory allocation.

### 4.2 `ModelRepository`

- **Methods**: `fetch(id, version) -> path`
- **Replaces**: The `asyncio.sleep(5)` simulation in `artifact_manager.py` that pretends to download models
- **Why**: Real implementation uses `huggingface_hub.snapshot_download()`. Tests inject `FakeModelRepository` returning a local path instantly.

### 4.3 `GPUTransfer`

- **Methods**: `copy_hbm_to_dram`, `copy_dram_to_hbm`
- **Replaces**: Direct GPU memory transfer calls that are already pure simulation in the current codebase
- **Why**: Allows full cache lifecycle testing without a GPU. `FakeGPUTransfer` returns success immediately.

---

## 5. Shared Types and Duplication Elimination

A `shared/types.py` module provides single definitions for types that were duplicated 3x with different shapes across `cache_manager.py`, `connector.py`, `controller.py`, and `gpu_transfer.py`:

| Type | Fields | Previously Duplicated In |
|------|--------|--------------------------|
| `BlockReference` | `device_id`, `memory_address`, `size_bytes` | cache_manager.py, connector.py, controller.py |
| `StorageNodeInfo` | `node_id`, `host`, `port` | connector.py, controller.py |
| `TransferResult` | `success`, `message` | gpu_transfer.py, cache_manager.py |
| `AllocationPointer` | `cpu_address`, `size_bytes` | l1_cache allocator, cache_manager.py |

---

## 6. Telemetry Metrics Specification

Implementation: `prometheus_client` library. Each FastAPI service gets `/metrics` via manual route or `prometheus_fastapi_instrumentator`.

### Gateway

| Metric | Type | Labels |
|--------|------|--------|
| `gateway_requests_total` | Counter | `model`, `status_code` |
| `gateway_request_duration_seconds` | Histogram | `model` |
| `gateway_routing_decisions_total` | Counter | `model`, `decision` (cache_hit/load_aware/fallback) |
| `gateway_active_connections` | Gauge | — |
| `gateway_backend_errors_total` | Counter | `model`, `error_type` |

### Engine

| Metric | Type | Labels |
|--------|------|--------|
| `engine_requests_total` | Counter | `model`, `status` |
| `engine_request_duration_seconds` | Histogram | `model` |
| `engine_time_to_first_token_seconds` | Histogram | `model` |
| `engine_tokens_generated_total` | Counter | `model` |
| `engine_tokens_per_second` | Gauge | `model` |
| `engine_batch_size` | Histogram | `model` |
| `engine_pending_requests` | Gauge | `model` |
| `engine_gpu_memory_used_bytes` | Gauge | `device` |
| `engine_lora_load_duration_seconds` | Histogram | `adapter` |
| `engine_lora_active` | Gauge | — |
| `engine_stream_cancelled_total` | Counter | `model` |

**Alert threshold**: GPU memory at 90% of 8GB budget.

### Sidecar

| Metric | Type | Labels |
|--------|------|--------|
| `sidecar_model_load_duration_seconds` | Histogram | `model`, `source` |
| `sidecar_adapter_load_duration_seconds` | Histogram | `adapter` |
| `sidecar_resident_models` | Gauge | — |
| `sidecar_resident_adapters` | Gauge | — |
| `sidecar_download_bytes_total` | Counter | `artifact_type` |

### Cache (L1 + L2)

| Metric | Type | Labels |
|--------|------|--------|
| `cache_l1_capacity_bytes` | Gauge | — |
| `cache_l1_used_bytes` | Gauge | — |
| `cache_l1_hits_total` | Counter | — |
| `cache_l1_misses_total` | Counter | — |
| `cache_l1_evictions_total` | Counter | `reason` |
| `cache_l2_hits_total` | Counter | — |
| `cache_l2_misses_total` | Counter | — |
| `cache_l2_put_duration_seconds` | Histogram | — |
| `cache_l2_get_duration_seconds` | Histogram | — |
| `cache_l2_active_nodes` | Gauge | — |

---

## 7. Local Testing Strategy

### docker-compose services

| Service | Port | Image | GPU? |
|---------|------|-------|------|
| `gateway` | 8000 | `docker/gateway.dockerfile` | No |
| `engine` | 8080 | `docker/inference.dockerfile` | Yes |
| `sidecar` | 8001 | `docker/sidecar.dockerfile` | No |
| `redis` | 6379 | `redis:7-alpine` | No |
| `prometheus` | 9090 | `prom/prometheus` | No |

- Engine + sidecar share `/mnt/models` volume
- Small model (`Qwen/Qwen2-0.5B`) loads in <30s
- Single Redis instance for L2
- `.env` file for secrets (HF token, etc.)

### Test framework

- `pytest` + `pytest-asyncio` + `httpx` (async TestClient)
- `tests/unit/` — no Docker, mock all I/O
- `tests/integration/` — requires docker-compose
- `conftest.py` — shared fixtures (mock engine, TestClients)

### Mock strategy

When `ENABLE_ENGINE_MOCK=true`, the engine uses `MockLLMEngine` with deterministic token generation. All other subsystems (metrics, routing, batching, streaming, error handling) remain fully functional. This allows complete system testing on machines without GPUs.
