# Infrastructure Overview

A distributed LLM serving platform (inspired by AIBrix) designed for
multi-model, multi-LoRA inference on Kubernetes. The system pairs a lightweight
gateway with N inference pods, each running a vLLM-based engine alongside a
model-management sidecar, backed by a tiered KV cache.

## Status Legend

| Tag | Meaning |
|-----|---------|
| `[LIVE]` | Implemented and functional |
| `[STUB]` | Code exists but uses mocks / simulated I/O |
| `[PLANNED]` | Comment or empty file only |

---

## 1. High-Level Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              Control Plane [PLANNED]         │
                          │  Autoscaler ─ LoRA Mgr ─ Admission Ctrl     │
                          └──────────────┬─────────────────────────────┘
                                         │ watches / scales
                                         ▼
  ┌────────┐  POST /generate   ┌─────────────────┐
  │ Client ├──────────────────►│  Gateway :8000   │ [LIVE]
  └────────┘                   │  (static route)  │
                               └────────┬────────┘
                                        │ routes by model name
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │ Pod A    │  │ Pod B    │  │ Pod N    │
                   │ Engine   │  │ Engine   │  │ Engine   │
                   │ +Sidecar │  │ +Sidecar │  │ +Sidecar │
                   └────┬─────┘  └────┬─────┘  └────┬─────┘
                        │             │             │
                        └──────┬──────┘─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Redis L2 Cache      │ [STUB]
                    │  (consistent hash)   │
                    └──────────────────────┘
```

---

## 2. Pod Architecture (Engine + Sidecar)

Each inference pod runs two containers sharing a `/mnt/models` volume:

```
┌─────────────────────────────────────────────────────────┐
│  Inference Pod                                          │
│                                                         │
│  ┌─────────────────────────┐  ┌──────────────────────┐  │
│  │  Engine  :8080 [LIVE]   │  │  Sidecar :8001 [STUB]│  │
│  │                         │  │                       │  │
│  │  vLLM LLMEngine        │  │  ArtifactManager      │  │
│  │  continuous batching    │  │  model registry       │  │
│  │  futures-based queue    │  │  adapter registry     │  │
│  │  LoRA hot-loading       │  │  L1 cache (DRAM)      │  │
│  │  mock mode available    │  │  L2 connector (Redis) │  │
│  │                         │  │                       │  │
│  │  /health  /ready        │  │  /health  /ready      │  │
│  │  /inference  /metrics   │  │  /load  /unload       │  │
│  │                         │  │  /registry            │  │
│  │                         │  │  /adapter/fetch       │  │
│  └────────────┬────────────┘  └───────────┬──────────┘  │
│               │         shared volume      │            │
│               └──────── /mnt/models ───────┘            │
└─────────────────────────────────────────────────────────┘
```

### Engine (:8080) `[LIVE]`

The engine wraps vLLM's `LLMEngine` with a FastAPI server. Key behaviours:

- **Continuous batching loop** &mdash; async loop calling `engine.step()`,
  resolving `asyncio.Future` per request (`engine.py:102-127`).
- **Futures-based queue** &mdash; `add_request()` returns an awaitable future;
  callers block until their result is ready (`engine.py:54-96`).
- **LoRA hot-loading** &mdash; on adapter request, calls sidecar
  `/adapter/fetch/{id}`, then issues `LoRARequest` into vLLM
  (`engine.py:76-92`).
- **Mock mode** &mdash; set `ENABLE_ENGINE_MOCK=true` to swap in
  `MockLLMEngine` (deterministic responses, no GPU) (`mock_engine.py`).
- **Input validation** &mdash; Pydantic model with bounds on `max_tokens`
  (1-4096), `temperature` (0.0-2.0), queue cap (`max_pending`, default 10)
  (`api.py:19-25`, `api.py:129-134`).

### Sidecar (:8001) `[STUB]`

Manages model and adapter artifacts on the shared volume:

- **ArtifactManager** &mdash; downloads models/adapters from remote storage
  (currently simulated with `asyncio.sleep`) to `/mnt/models`
  (`artifact_manager.py:28-47`).
- **Model registry** &mdash; dict mapping `model_identifier -> local_path`;
  blocks readiness until the initial model is loaded (`api.py:9-25`).
- **Adapter registry** &mdash; dict mapping `adapter_identifier -> local_path`;
  max 10 resident adapters per pod (`artifact_manager.py:22`).
- **L1/L2 cache management** &mdash; L1CacheAPI and L2Connector are wired into
  the `MultiTieredCacheManager` (see Section 4).

---

## 3. Gateway `[LIVE]`

### Current: Static Routing

The gateway uses a hardcoded `MODEL_SERVICE_MAP` dict to route
`POST /generate` requests to the correct inference pod by model name
(`routing.py:32-35`):

```python
MODEL_SERVICE_MAP = {
    "llama-3-8b": "http://vllm-llama-8b-svc:8000",
    "mistral-7b": "http://vllm-mistral-7b-svc:8000",
}
```

An `httpx.AsyncClient` (5-min timeout) forwards the request JSON to
`{worker_url}/inference`.

### Planned: Load-Aware Routing `[PLANNED]`

The `routing.py` TODO notes the intent to route based on queue length,
KV cache hit ratio, and GPU utilisation. This is not yet implemented.

### Request Flow

```
Client                Gateway :8000             Engine :8080            Sidecar :8001
  │                       │                         │                       │
  │  POST /generate       │                         │                       │
  │  {model, prompt, ...} │                         │                       │
  │──────────────────────►│                         │                       │
  │                       │  POST /inference        │                       │
  │                       │─────────────────────────►                       │
  │                       │                         │                       │
  │                       │                         │  (if adapter needed)  │
  │                       │                         │  POST /adapter/fetch  │
  │                       │                         │──────────────────────►│
  │                       │                         │  {local_path}         │
  │                       │                         │◄──────────────────────│
  │                       │                         │                       │
  │                       │                         │  LoRARequest→vLLM     │
  │                       │                         │  continuous_batch_loop│
  │                       │                         │  resolves future      │
  │                       │  {text, tokens, dur}    │                       │
  │                       │◄─────────────────────────                       │
  │  {text, tokens, dur}  │                         │                       │
  │◄──────────────────────│                         │                       │
```

---

## 4. KV Cache Architecture (3 Tiers)

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Tier 0: GPU HBM                      [LIVE via vLLM]               │
│  ┌─────────────────────────────────┐                                 │
│  │  vLLM PagedAttention            │                                 │
│  │  (automatic paged KV mgmt)      │                                 │
│  └───────────────┬─────────────────┘                                 │
│                  │ async CUDA memcpy (simulated)                     │
│                  ▼                                                    │
│  Tier 1: L1 CPU DRAM                  [STUB]                        │
│  ┌─────────────────────────────────┐                                 │
│  │  L1CacheAPI                     │                                 │
│  │  ├── L1Allocator (pinned pool)  │  capacity: 512 MB (default)    │
│  │  ├── LRU EvictionPolicy         │  evicts least-recently-used    │
│  │  └── GPUTransferHandler         │  simulated cudaMemcpyAsync     │
│  └───────────────┬─────────────────┘                                 │
│                  │ bytes over Redis client                           │
│                  ▼                                                    │
│  Tier 2: L2 Redis                      [STUB]                       │
│  ┌─────────────────────────────────┐                                 │
│  │  L2Connector                    │                                 │
│  │  ├── ConsistentHashRing         │  SHA-256, 3 virtual nodes/node │
│  │  ├── KVCacheWatcherClient       │  gRPC topology updates         │
│  │  └── redis.asyncio pool         │  per-node connection pool      │
│  └─────────────────────────────────┘                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### MultiTieredCacheManager (`cache_manager.py`)

Orchestrates offload and fetch across tiers:

- **Offload** &mdash; try L1 first; if L1 is full (eviction didn't help),
  promote to L2 (`cache_manager.py:25-52`).
- **Fetch** &mdash; check L1 first; on miss, check L2 and restore to HBM
  (`cache_manager.py:54-73`).
- `check_global_availability` and `execute_l1_eviction` are declared but raise
  `NotImplementedError`.

### L1 Components

| File | Role |
|------|------|
| `l1_cache/allocator.py` | Fixed-size pinned DRAM pool; simulated address math |
| `l1_cache/eviction_policy.py` | LRU via `OrderedDict` |
| `l1_cache/gpu_transfer.py` | Simulated async CUDA memcpy (0.001s sleep) |
| `l1_cache/api.py` | Facade: allocate &rarr; transfer &rarr; evict loop |

### L2 Components

| File | Role |
|------|------|
| `l2_cache/connector.py` | `ConsistentHashRing` + `KVCacheWatcherClient` + Redis pool |
| `distributed_cache/controller.py` | `KVCacheWatcher` gRPC servicer (mock topology + health loop) |
| `distributed_cache/storage.py` | `StorageNode` wrapping a single Redis instance |

---

## 5. Model & LoRA Lifecycle

### Base Model Loading

1. Sidecar starts &rarr; `check_initial_model_load()` runs as a background task.
2. `ArtifactManager.load_model()` downloads model to `/mnt/models/{id}/{version}`
   (currently simulated).
3. `is_ready` flips to `True` &rarr; readiness probe passes.
4. Engine initialises `LLMEngine` with the model from disk.

### LoRA Adapter Loading

1. Client sends request with `adapter_identifier` + `adapter_version`.
2. Engine calls sidecar `POST /adapter/fetch/{adapter_identifier}`.
3. Sidecar downloads adapter weights to shared volume, returns `local_path`.
4. Engine issues `LoRARequest` into vLLM (`engine.py:130-144`).

```
                ┌─────────────┐      ┌────────────┐      ┌──────────┐
                │ NOT_PRESENT │─────►│  ON_DISK   │─────►│ IN_VRAM  │
                │             │fetch │ /mnt/models│load  │  (vLLM)  │
                └─────────────┘      └────────────┘      └──────────┘

                Max 10 resident adapters per pod (VRAM constraint)
```

---

## 6. Control Plane `[PLANNED]`

All control plane components are scaffolds (1-3 line files):

| Component | File | Intended Purpose |
|-----------|------|-----------------|
| **Autoscaler** | `control_plane/autoscaler.py` | LLM-aware pod scaling based on queue length; cold-start mitigation via minimum replicas |
| **LoRA Manager** | `control_plane/lora_manager.py` | Cluster-wide LoRA placement and lifecycle orchestration |
| **Admission Controller** | `control_plane/admission_controller.py` | Kubernetes policy enforcement for inference workloads |

---

## 7. Kubernetes Deployment Model

### Custom Resource Definitions

| CRD | File | Status |
|-----|------|--------|
| `LoraAdapter` | `kubernetes/crds/LoraAdapter.yaml` | `[PLANNED]` (empty) |
| `RayClusterFleet` | `kubernetes/crds/RayClusterFleet.yaml` | `[PLANNED]` (empty) |

### Manifests

| File | Purpose | Status |
|------|---------|--------|
| `manifests/01-rbac.yaml` | RBAC for controller and LoRA Manager | `[PLANNED]` (comment only) |
| `manifests/02-crd-lora-adapter.yaml` | LoraAdapter CRD definition | `[PLANNED]` (comment only) |
| `manifests/03-controller-deploy.yaml` | Controller deployment | `[PLANNED]` (scaffold) |
| `manifests/04-gateway-deploy.yaml` | Gateway deployment | `[PLANNED]` (scaffold) |
| `manifests/05-inference-service-deploy.yaml` | Inference pod deployment | `[PLANNED]` (scaffold) |

### Dockerfiles

All four Dockerfiles are empty placeholders:

| File | Target |
|------|--------|
| `docker/inference.dockerfile` | Engine container |
| `docker/sidecar.dockerfile` | Sidecar container |
| `docker/gateway.dockerfile` | Gateway container |
| `docker/controller.dockerfile` | Control plane controller |

### Target Topology

```
┌─────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                             │
│                                                                 │
│  ┌───────────────────┐     ┌──────────────────────────────────┐ │
│  │  Controller Pod    │     │  Gateway Pod                    │ │
│  │  [PLANNED]         │     │  routing.py :8000   [LIVE]      │ │
│  │  autoscaler        │     └───────────┬────────────────────┘ │
│  │  lora_manager      │                 │                      │
│  │  admission_ctrl    │                 ▼                      │
│  └───────────────────┘     ┌────────────────────────────┐      │
│                            │  Inference Pods (N replicas)│      │
│                            │  ┌────────┐  ┌──────────┐  │      │
│                            │  │Engine  │  │ Sidecar  │  │      │
│                            │  │ :8080  │  │  :8001   │  │      │
│                            │  └────┬───┘  └────┬─────┘  │      │
│                            │       └─/mnt/models┘       │      │
│                            └────────────┬───────────────┘      │
│                                         │                      │
│                            ┌────────────▼───────────────┐      │
│                            │  Redis StatefulSet          │      │
│                            │  redis-0, redis-1, redis-2  │      │
│                            │  [STUB]                     │      │
│                            └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Metrics & Observability

### Engine Metrics `[LIVE]`

All metrics are defined in `data_plane/inference/engine/metrics.py`. The
"Wired" column indicates whether the metric is actively recorded in `api.py`.

| Metric | Type | Labels | Wired |
|--------|------|--------|-------|
| `engine_requests_total` | Counter | `model`, `status` | Yes |
| `engine_request_duration_seconds` | Histogram | `model` | Yes |
| `engine_time_to_first_token_seconds` | Histogram | `model` | No |
| `engine_tokens_generated_total` | Counter | `model` | Yes |
| `engine_tokens_per_second` | Gauge | `model` | No |
| `engine_batch_size` | Histogram | `model` | No |
| `engine_pending_requests` | Gauge | `model` | No |
| `engine_gpu_memory_used_bytes` | Gauge | `device` | No |
| `engine_lora_load_duration_seconds` | Histogram | `adapter` | No |
| `engine_lora_active` | Gauge | &mdash; | No |
| `engine_stream_cancelled_total` | Counter | `model` | No |

Prometheus endpoint: `GET /metrics` (engine `api.py:184-187`).

### Gateway Metrics `[PLANNED]`

No metrics are defined for the gateway yet.

### Cache Metrics `[PLANNED]`

No metrics are defined for L1/L2 cache operations yet.

---

## 9. Configuration Reference

All config classes use `pydantic-settings` with env-var binding.

### EngineConfig

Source: `data_plane/inference/engine/config.py`

| Env Var | Type | Default | Description |
|---------|------|---------|-------------|
| `ENGINE_MODEL_NAME` | `str` | `Qwen/Qwen2-0.5B` | HuggingFace model ID |
| `ENGINE_MAX_MODEL_LEN` | `int` | `512` | Maximum sequence length |
| `ENGINE_GPU_MEMORY_UTILIZATION` | `float` | `0.70` | Fraction of GPU memory for KV cache |
| `ENGINE_DTYPE` | `str` | `bfloat16` | Model weight data type |
| `ENGINE_HOST` | `str` | `0.0.0.0` | Bind host |
| `ENGINE_PORT` | `int` | `8080` | Bind port |
| `ENGINE_SIDECAR_URL` | `str` | `http://localhost:8001` | Sidecar base URL |
| `ENGINE_MODEL_PATH` | `str` | `/models/resident_model` | Local model file path |
| `ENGINE_ENABLE_LORA` | `bool` | `false` | Enable vLLM LoRA support |
| `ENGINE_MAX_PENDING` | `int` | `10` | Max queued requests before 429 |
| `ENGINE_TEMPERATURE` | `float` | `0.0` | Default sampling temperature |
| `ENABLE_ENGINE_MOCK` | `bool` | `false` | Use mock engine (no GPU) |

### SidecarConfig

Source: `data_plane/inference/sidecar/config.py`

| Env Var | Type | Default | Description |
|---------|------|---------|-------------|
| `SIDECAR_HOST` | `str` | `0.0.0.0` | Bind host |
| `SIDECAR_PORT` | `int` | `8001` | Bind port |
| `SIDECAR_SHARED_VOLUME` | `str` | `/mnt/models` | Shared volume mount path |
| `SIDECAR_MAX_ADAPTERS` | `int` | `10` | Max resident LoRA adapters |
| `SIDECAR_L1_CAPACITY_MB` | `int` | `512` | L1 DRAM cache size in MB |
| `SIDECAR_ENGINE_URL` | `str` | `http://localhost:8080` | Engine base URL |
| `SIDECAR_L2_REDIS_HOST` | `str` | `localhost` | Redis host for L2 cache |
| `SIDECAR_L2_REDIS_PORT` | `int` | `6379` | Redis port for L2 cache |
| `SIDECAR_GRPC_PORT` | `int` | `50051` | gRPC port for topology updates |
| `SIDECAR_INITIAL_MODEL` | `str` | `default-llama-3` | Model to load on startup |
| `SIDECAR_INITIAL_MODEL_VERSION` | `str` | `v1.0` | Initial model version |
| `SIDECAR_REGISTRY_PATH` | `str` | `/mnt/models/registry.json` | Persistent registry file |

### GatewayConfig

Source: `data_plane/gateway/config.py`

| Env Var | Type | Default | Description |
|---------|------|---------|-------------|
| `GATEWAY_HOST` | `str` | `0.0.0.0` | Bind host |
| `GATEWAY_PORT` | `int` | `8000` | Bind port |
| `GATEWAY_REQUEST_TIMEOUT` | `float` | `300.0` | Request timeout in seconds |

---

## 10. Project Structure

```
inference-server/
├── data_plane/
│   ├── gateway/
│   │   ├── routing.py              # FastAPI gateway, static MODEL_SERVICE_MAP [LIVE]
│   │   └── config.py               # GatewayConfig (pydantic-settings)       [LIVE]
│   └── inference/
│       ├── engine/
│       │   ├── api.py              # FastAPI server: /health /ready /inference /metrics [LIVE]
│       │   ├── engine.py           # vLLM Engine wrapper, batching loop, LoRA  [LIVE]
│       │   ├── mock_engine.py      # MockLLMEngine for GPU-free testing        [LIVE]
│       │   ├── config.py           # EngineConfig (pydantic-settings)          [LIVE]
│       │   └── metrics.py          # 11 Prometheus metrics (3 wired)           [LIVE]
│       ├── sidecar/
│       │   ├── api.py              # FastAPI server: /health /ready /load etc  [STUB]
│       │   ├── artifact_manager.py # Model & adapter download + registry       [STUB]
│       │   ├── config.py           # SidecarConfig (pydantic-settings)         [LIVE]
│       │   ├── cache_manager.py    # MultiTieredCacheManager (L1→L2 fallback)  [STUB]
│       │   ├── kv_cache_api.py     # KV cache API surface                      [STUB]
│       │   ├── l1_cache/
│       │   │   ├── allocator.py    # Pinned DRAM pool allocator                [STUB]
│       │   │   ├── eviction_policy.py # LRU eviction via OrderedDict           [STUB]
│       │   │   ├── gpu_transfer.py # Simulated CUDA memcpy                     [STUB]
│       │   │   └── api.py          # L1CacheAPI facade                         [STUB]
│       │   └── l2_cache/
│       │       ├── connector.py    # ConsistentHashRing + L2Connector          [STUB]
│       │       └── api.py          # L2 cache API surface                      [STUB]
│       └── distributed_cache/
│           ├── controller.py       # KVCacheWatcher gRPC servicer              [STUB]
│           └── storage.py          # StorageNode (Redis wrapper)               [STUB]
├── control_plane/
│   ├── autoscaler.py               # LLM-aware scaling                         [PLANNED]
│   ├── lora_manager.py             # Cluster LoRA orchestration                [PLANNED]
│   └── admission_controller.py     # K8s policy enforcement                    [PLANNED]
├── shared/
│   ├── types.py                    # BlockReference, TransferResult, etc.      [LIVE]
│   ├── ports.py                    # Protocol interfaces (CacheStore, etc.)    [LIVE]
│   └── proto/                      # gRPC proto definitions                    [PLANNED]
├── docker/
│   ├── inference.dockerfile        # Engine container image                    [PLANNED]
│   ├── sidecar.dockerfile          # Sidecar container image                   [PLANNED]
│   ├── gateway.dockerfile          # Gateway container image                   [PLANNED]
│   └── controller.dockerfile       # Controller container image                [PLANNED]
├── manifests/
│   ├── 01-rbac.yaml                # RBAC rules                                [PLANNED]
│   ├── 02-crd-lora-adapter.yaml    # LoraAdapter CRD                           [PLANNED]
│   ├── 03-controller-deploy.yaml   # Controller deployment                     [PLANNED]
│   ├── 04-gateway-deploy.yaml      # Gateway deployment                        [PLANNED]
│   └── 05-inference-service-deploy.yaml # Inference pod deployment             [PLANNED]
├── kubernetes/
│   └── crds/
│       ├── LoraAdapter.yaml        # LoraAdapter CRD schema                    [PLANNED]
│       └── RayClusterFleet.yaml    # RayClusterFleet CRD schema                [PLANNED]
└── tests/
    ├── conftest.py                 # Shared fixtures                            [LIVE]
    ├── unit/
    │   ├── test_smoke.py           # Import smoke tests                        [LIVE]
    │   └── test_engine.py          # Engine unit tests                         [LIVE]
    ├── integration/                #                                            [PLANNED]
    └── fakes/                      #                                            [PLANNED]
```
