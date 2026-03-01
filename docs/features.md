# Features & Architecture Deep Dive

This document is the single entry point for understanding the inference server. It covers the system lifecycle, each component's internals, every API endpoint, and hands-on examples for testing.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Lifecycle](#lifecycle)
   - [Cold Start](#cold-start)
   - [Steady State](#steady-state)
   - [Shutdown](#shutdown)
3. [Component Deep Dives](#component-deep-dives)
   - [Gateway](#1-gateway-port-8000)
   - [Engine](#2-engine-port-8080)
   - [Sidecar](#3-sidecar-port-8001)
4. [Supporting Infrastructure](#supporting-infrastructure)
5. [Inter-Service Communication](#inter-service-communication)
6. [Configuration Reference](#configuration-reference)
7. [Testing Guide](#testing-guide)
   - [Running the Stack](#running-the-stack)
   - [Component-Level Tests](#component-level-tests)
   - [End-to-End Tests](#end-to-end-tests)
   - [Monitoring & Metrics](#monitoring--metrics)

---

## System Overview

The inference server is a Kubernetes-native LLM serving platform split into three containers that run together in a single pod:

```
                 ┌──────────────────────────────────────────────┐
  Client ──────▶ │  Gateway (:8000)                             │
                 │    routes /generate → engine by model name   │
                 └──────────┬───────────────────────────────────┘
                            │ POST /inference
                            ▼
                 ┌──────────────────────────────────────────────┐
                 │  Engine (:8080)                              │
                 │    vLLM runtime · continuous batching         │
                 │    LoRA adapter hot-loading                   │
                 │        │                                     │
                 │        │ GET /registry/models (poll)          │
                 │        │ POST /adapter/fetch/{id}            │
                 │        ▼                                     │
                 │  Sidecar (:8001)                             │
                 │    model download (HuggingFace Hub)           │
                 │    adapter management · artifact registry     │
                 │        │                                     │
                 │        ▼                                     │
                 │  /mnt/models (shared volume)                 │
                 └──────────────────────────────────────────────┘
                 │  Redis (:6379) ← L2 KV cache                 │
                 │  Prometheus (:9090) ← metrics scraping        │
                 └──────────────────────────────────────────────┘
```

| Container | Framework | Port | Responsibility |
|-----------|-----------|------|----------------|
| Gateway | FastAPI | 8000 | Request routing, model-to-service mapping |
| Engine | FastAPI + vLLM | 8080 | GPU inference, continuous batching, LoRA adapters |
| Sidecar | FastAPI | 8001 (HTTP), 50051 (gRPC) | Model/adapter downloads, artifact registry |
| Redis | redis:7-alpine | 6379 | L2 KV cache backend |
| Prometheus | prom/prometheus | 9090 | Metrics collection |

---

## Lifecycle

### Cold Start

The startup sequence is orchestrated through container dependencies and polling:

```
1. Redis starts
       ↓
2. Sidecar starts
   ├── ArtifactManager initializes
   ├── Registers initial model as status="downloading"
   ├── Background task: downloads model from HuggingFace Hub
   │   └── snapshot_download() → /mnt/models/{model}/
   ├── On success: status="loaded", is_ready=True
   └── /health returns 200, /ready returns 200
       ↓
3. Engine starts
   ├── FastAPI boots, /health returns 503 (initializing)
   ├── Background task: _init_engine()
   │   ├── Polls GET sidecar:8001/registry/models every 2s
   │   ├── Waits until model status="loaded" (up to 600s timeout)
   │   ├── Gets local_path from sidecar registry
   │   ├── Initializes vLLM LLMEngine with model weights
   │   ├── Starts continuous_batching_loop()
   │   └── /health returns 200, /ready returns 200
       ↓
4. Gateway starts
   ├── Creates httpx.AsyncClient (300s timeout)
   ├── /health returns 200 immediately
   └── Routes requests to engine based on MODEL_SERVICE_MAP
       ↓
5. Prometheus starts
   └── Scrapes /metrics from all three services
```

**Key timing**: The sidecar model download can take minutes for large models. The engine will poll the sidecar registry every `ENGINE_SIDECAR_POLL_INTERVAL` seconds (default: 2s) and wait up to `ENGINE_SIDECAR_TIMEOUT` seconds (default: 600s) before giving up.

### Steady State

Once all containers are ready, the inference flow is:

```
Client
  │
  │  POST /generate {"model":"llama-3-8b","prompt":"Hello","max_tokens":256}
  ▼
Gateway (:8000)
  │  1. Looks up model in MODEL_SERVICE_MAP
  │  2. Forwards full request body to engine
  │
  │  POST /inference {"prompt":"Hello","max_tokens":256}
  ▼
Engine (:8080)
  │  3. Validates request (prompt length, token limits, temperature)
  │  4. Checks queue capacity (max_pending)
  │  5. If adapter requested: fetches via sidecar
  │  6. Submits to vLLM engine, awaits result from batching loop
  │  7. Returns InferenceResponse
  ▼
Response: {"text":"...","tokens_generated":42,"duration_seconds":1.23}
```

**Continuous batching loop**: The engine runs a tight loop (`continuous_batching_loop`) that calls `engine.step()` to process requests in batches. Requests are added via `add_request()` which creates a Future that the batching loop resolves when generation completes.

### Shutdown

On SIGTERM (or `docker-compose down`):

1. **Engine**: Cancels `_init_task` (if still running), cancels `_batching_loop`, closes
2. **Gateway**: Closes httpx client
3. **Sidecar**: Logs shutdown (no special cleanup)

---

## Component Deep Dives

### 1. Gateway (Port 8000)

**Source**: `data_plane/gateway/routing.py`

The gateway is a thin routing layer. It receives client requests, maps the model name to a backend engine URL, and forwards the request.

#### Routing Table

The model-to-service mapping is currently hardcoded:

```python
MODEL_SERVICE_MAP = {
    "llama-3-8b": "http://engine:8080",
}
```

To add models, add entries to this dict. The gateway forwards to `{worker_url}/inference`.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Always returns `{"status": "healthy"}` |
| `POST` | `/generate` | Routes inference request to the appropriate engine |

##### `GET /health`

Simple liveness check. Always returns 200.

```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy"}
```

##### `POST /generate`

Accepts a JSON body with at least a `model` field. The entire body is forwarded to the engine's `/inference` endpoint.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "prompt": "Explain quantum computing in one sentence",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Success (200)**:
```json
{
  "text": "Quantum computing uses quantum bits...",
  "tokens_generated": 15,
  "duration_seconds": 1.23
}
```

**Model not found (404)**:
```json
{"detail": "Model unknown-model not found"}
```

**Engine unreachable (503)**:
```json
{"detail": "Model service 'llama-3-8b' is unreachable."}
```

#### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `GATEWAY_PORT` | `8000` | Bind port |
| `GATEWAY_REQUEST_TIMEOUT` | `300.0` | httpx client timeout (seconds) |

---

### 2. Engine (Port 8080)

**Source**: `data_plane/inference/engine/api.py`, `engine.py`, `mock_engine.py`, `metrics.py`

The engine is the core inference service. It wraps vLLM's `LLMEngine` in an async FastAPI server with continuous batching, LoRA adapter hot-loading, and Prometheus metrics.

#### Modes of Operation

| Mode | Env Var | Description |
|------|---------|-------------|
| **Real (vLLM)** | `ENABLE_ENGINE_MOCK=false` (default) | Requires GPU. Waits for sidecar to load model, initializes vLLM. |
| **Mock** | `ENABLE_ENGINE_MOCK=true` | No GPU needed. Returns deterministic responses. Useful for development and testing. |

**Mock engine responses** (keyword-based):

| Prompt contains | Response |
|-----------------|----------|
| `"hello"` | `"Hello! How can I help you today?"` |
| `"who"` | `"I'm an AI assistant created by Anthropic."` |
| `"what"` | `"I'm a language model trained to be helpful, harmless, and honest."` |
| `"test"` | `"This is a test response from the mock engine."` |
| *(anything else)* | `"The quick brown fox jumps over the lazy dog. This is a mock response."` |

#### Request Model

```python
class InferenceRequest(BaseModel):
    prompt: str           # Required, min_length=1
    max_tokens: int       # Default: 256, range: [1, 4096]
    temperature: float    # Default: 0.7, range: [0.0, 2.0]
    stream: bool          # Default: False
    adapter_identifier: str | None  # Optional LoRA adapter ID
    adapter_version: str | None     # Optional adapter version
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns 200 when engine is initialized, 503 during startup |
| `GET` | `/ready` | Returns 200 when engine is initialized AND model is loaded |
| `POST` | `/inference` | Run inference on a prompt |
| `GET` | `/metrics` | Prometheus metrics |

##### `GET /health`

```bash
curl http://localhost:8080/health
```

**During initialization (503)**:
```json
{"status": "initializing"}
```

**After engine ready (200)**:
```json
{"status": "healthy"}
```

##### `GET /ready`

```bash
curl http://localhost:8080/ready
```

**Not ready (503)**:
```json
{"status": "not_ready", "reason": "engine_not_initialized"}
```
or
```json
{"status": "not_ready", "reason": "model_loading"}
```

**Ready (200)**:
```json
{"status": "ready"}
```

##### `POST /inference`

```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.3
  }'
```

**Success (200)**:
```json
{
  "text": "The capital of France is Paris...",
  "tokens_generated": 12,
  "duration_seconds": 0.87
}
```

**Queue full (429)**:
```json
{"detail": "Queue full, too many pending requests"}
```
Headers: `Retry-After: 1`

**Not ready (503)**:
```json
{"detail": "Engine not initialized"}
```

**Timeout (504)**:
```json
{"detail": "Request generation timed out"}
```

##### `POST /inference` with LoRA Adapter

When an adapter is requested, the engine fetches it from the sidecar before running inference:

```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to French: Hello world",
    "max_tokens": 50,
    "adapter_identifier": "my-org/translation-lora",
    "adapter_version": "v1.0"
  }'
```

Under the hood:
1. Engine calls `POST sidecar:8001/adapter/fetch/my-org/translation-lora?version=v1.0`
2. Sidecar downloads adapter from HuggingFace if not cached
3. Engine calls `vllm.add_lora()` with the local adapter path
4. Inference runs with the adapter applied

##### `GET /metrics`

```bash
curl http://localhost:8080/metrics
```

Returns Prometheus text format with these metrics:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `engine_requests_total` | Counter | `model`, `status` | Total requests (status: success/timeout/error) |
| `engine_request_duration_seconds` | Histogram | `model` | Request latency |
| `engine_time_to_first_token_seconds` | Histogram | `model` | TTFT latency |
| `engine_tokens_generated_total` | Counter | `model` | Total tokens produced |
| `engine_tokens_per_second` | Gauge | `model` | Current throughput |
| `engine_batch_size` | Histogram | `model` | Batch sizes (buckets: 1,2,4,8,16,32) |
| `engine_pending_requests` | Gauge | `model` | Queue depth |
| `engine_gpu_memory_used_bytes` | Gauge | `device` | GPU memory usage |
| `engine_lora_load_duration_seconds` | Histogram | `adapter` | LoRA loading time |
| `engine_lora_active` | Gauge | - | Active LoRA count |
| `engine_stream_cancelled_total` | Counter | `model` | Cancelled streams |

#### Continuous Batching Loop

The engine runs a permanent async loop:

```
loop:
  if no unfinished requests → sleep 10ms → continue
  outputs = engine.step()          # vLLM processes one batch
  for each output:
    if finished → resolve the request's Future with generated text
```

This allows multiple concurrent requests to be batched together by vLLM for efficient GPU utilization.

#### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ENGINE_MODEL_NAME` | `Qwen/Qwen2-0.5B` | HuggingFace model ID |
| `ENGINE_MAX_MODEL_LEN` | `512` | Max context length |
| `ENGINE_GPU_MEM_UTIL` | `0.70` | GPU memory fraction for model |
| `ENGINE_DTYPE` | `bfloat16` | Weight precision |
| `ENGINE_HOST` | `0.0.0.0` | Bind address |
| `ENGINE_PORT` | `8080` | Bind port |
| `ENGINE_SIDECAR_URL` | `http://localhost:8001` | Sidecar base URL |
| `ENGINE_SIDECAR_POLL_INTERVAL` | `2.0` | Poll interval for model readiness (seconds) |
| `ENGINE_SIDECAR_TIMEOUT` | `600.0` | Max wait for sidecar model load (seconds) |
| `ENGINE_MODEL_PATH` | `/models/resident_model` | Fallback model path |
| `ENGINE_ENABLE_LORA` | `false` | Enable LoRA adapter support in vLLM |
| `ENGINE_MAX_PENDING` | `10` | Max queued requests before 429 |
| `ENGINE_TEMPERATURE` | `0.0` | Default sampling temperature |
| `ENABLE_ENGINE_MOCK` | `false` | Use mock engine (no GPU) |

---

### 3. Sidecar (Port 8001)

**Source**: `data_plane/inference/sidecar/api.py`, `artifact_manager.py`, `cache_manager.py`, `metrics.py`

The sidecar manages model and adapter artifacts. It downloads models from HuggingFace Hub, stores them on a shared volume (`/mnt/models`), and maintains a registry of what's loaded. The engine polls this registry to know when models are ready.

#### ArtifactManager

The core class that handles all artifact operations:

- **Model registry**: `Dict[str, Dict]` mapping model identifiers to metadata (`model_id`, `version`, `status`, `local_path`)
- **Adapter registry**: `Dict[str, Dict]` mapping adapter identifiers to metadata
- **Persistence**: Registry state is saved to `registry.json` on the shared volume and restored on restart
- **Atomic downloads**: Models are downloaded to a temp directory, then atomically renamed into place to prevent partial files

**Download flow**:
```
snapshot_download(repo_id, revision)
  → downloads to /mnt/models/.dl-model-XXXXX/  (temp dir)
  → on success: rename to /mnt/models/{org}--{model}/{version}/
  → on failure: clean up temp dir
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns 200 when manager is initialized |
| `GET` | `/ready` | Returns 200 when initial model is loaded |
| `POST` | `/load/{model_identifier}` | Start loading a model (async) |
| `POST` | `/unload/{model_identifier}` | Remove a model from registry |
| `GET` | `/registry/models` | List all registered models with status |
| `GET` | `/registry/adapters` | List all registered adapters |
| `POST` | `/adapter/fetch/{adapter_identifier}` | Download/cache a LoRA adapter |
| `GET` | `/metrics` | Prometheus metrics |

##### `GET /health`

```bash
curl http://localhost:8001/health
```

**During init (503)**:
```json
{"status": "initializing"}
```

**Ready (200)**:
```json
{"status": "ok"}
```

##### `GET /ready`

```bash
curl http://localhost:8001/ready
```

**Not ready (503)** — initial model still downloading:
```json
{"detail": "Artifact Manager loading initial model."}
```

**Ready (200)**:
```json
{
  "status": "ready",
  "resident_models": ["arnir0/Tiny-LLM"]
}
```

##### `POST /load/{model_identifier}`

Triggers model download. The model identifier uses path-style encoding for namespaced models.

```bash
# Load a model (returns 202 immediately, downloads in background)
curl -X POST "http://localhost:8001/load/Qwen/Qwen2-0.5B?version=main"
```

**Download started (202)**:
```json
{"status": "downloading", "model_identifier": "Qwen/Qwen2-0.5B"}
```

**Already loaded (200)**:
```json
{
  "status": "loaded",
  "model_identifier": "Qwen/Qwen2-0.5B",
  "local_path": "/mnt/models/Qwen--Qwen2-0.5B/main"
}
```

**Already downloading (202)** — prevents duplicate downloads:
```json
{"status": "downloading", "model_identifier": "Qwen/Qwen2-0.5B"}
```

##### `POST /unload/{model_identifier}`

```bash
curl -X POST "http://localhost:8001/unload/Qwen/Qwen2-0.5B"
```

**Success (200)**:
```json
{"status": "success", "model_identifier": "Qwen/Qwen2-0.5B"}
```

**Not found (404)**:
```json
{"detail": "Model Qwen/Qwen2-0.5B not resident."}
```

##### `GET /registry/models`

Returns the full model registry. This is the endpoint the engine polls during startup.

```bash
curl http://localhost:8001/registry/models
```

**During download**:
```json
{
  "arnir0/Tiny-LLM": {
    "model_id": "arnir0/Tiny-LLM",
    "version": "main",
    "status": "downloading"
  }
}
```

**After download**:
```json
{
  "arnir0/Tiny-LLM": {
    "model_id": "arnir0/Tiny-LLM",
    "version": "main",
    "local_path": "/mnt/models/arnir0--Tiny-LLM/main",
    "status": "loaded"
  }
}
```

##### `GET /registry/adapters`

```bash
curl http://localhost:8001/registry/adapters
```

```json
{}
```

##### `POST /adapter/fetch/{adapter_identifier}`

Downloads and caches a LoRA adapter. This is called by the engine before inference when an adapter is requested.

```bash
curl -X POST "http://localhost:8001/adapter/fetch/my-org/my-lora?version=v1.0"
```

**Success (200)**:
```json
{
  "status": "success",
  "adapter_identifier": "my-org/my-lora",
  "local_path": "/mnt/models/my-org--my-lora/v1.0"
}
```

**Failure (500)**:
```json
{"detail": "Failed to fetch adapter: ..."}
```

##### `GET /metrics`

```bash
curl http://localhost:8001/metrics
```

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `sidecar_model_load_duration_seconds` | Histogram | `model`, `source` | Model download time |
| `sidecar_adapter_load_duration_seconds` | Histogram | `adapter` | Adapter download time |
| `sidecar_resident_models` | Gauge | - | Number of loaded models |
| `sidecar_resident_adapters` | Gauge | - | Number of loaded adapters |
| `sidecar_download_bytes_total` | Counter | `artifact_type` | Total bytes downloaded |

#### Multi-Tier KV Cache (cache_manager.py)

The sidecar includes a `MultiTieredCacheManager` for KV cache offloading:

- **L1 (DRAM)**: Fast local cache, configured via `SIDECAR_L1_CAPACITY_MB` (default: 512 MB)
- **L2 (Redis)**: Shared cache across pods, uses `allkeys-lru` eviction

```
Offload flow:  GPU HBM → try L1 (DRAM) → if full, promote to L2 (Redis)
Fetch flow:    check L1 → if miss, check L2 → if miss, cache miss (recompute)
```

#### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `SIDECAR_HOST` | `0.0.0.0` | Bind address |
| `SIDECAR_PORT` | `8001` | HTTP port |
| `SIDECAR_SHARED_VOLUME` | `/mnt/models` | Model storage path |
| `SIDECAR_MAX_ADAPTERS` | `10` | Max resident adapters |
| `SIDECAR_L1_CAPACITY_MB` | `512` | L1 cache size (MB) |
| `SIDECAR_ENGINE_URL` | `http://localhost:8080` | Engine URL |
| `SIDECAR_INITIAL_MODEL` | `Qwen/Qwen2.5-7B-Instruct-1M` | Model to download at startup |
| `SIDECAR_INITIAL_MODEL_VERSION` | `main` | Revision for initial model |
| `SIDECAR_REGISTRY_PATH` | `/mnt/models/registry.json` | Registry persistence path |
| `L2_REDIS_HOST` | `localhost` | Redis host |
| `L2_REDIS_PORT` | `6379` | Redis port |
| `GRPC_PORT` | `50051` | gRPC port |

---

## Supporting Infrastructure

### Redis (L2 Cache)

```yaml
redis:
  image: redis:7-alpine
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

Used for L2 KV cache. When the L1 DRAM cache is full, blocks are promoted to Redis. The `allkeys-lru` policy evicts least-recently-used keys when memory is full.

### Prometheus

Scrapes `/metrics` from all three services every 15 seconds (configurable in `docker/prometheus.yml`).

```bash
# Access Prometheus UI
open http://localhost:9090

# Example queries
# Request rate:
rate(engine_requests_total[5m])

# P99 latency:
histogram_quantile(0.99, rate(engine_request_duration_seconds_bucket[5m]))

# Resident models:
sidecar_resident_models
```

### Shared Volume

The engine and sidecar share `/mnt/models` (mapped to `./model-store` locally). The sidecar downloads artifacts here, and the engine reads model weights from here.

```
/mnt/models/
├── registry.json                    # Persisted registry state
├── arnir0--Tiny-LLM/
│   └── main/
│       ├── config.json
│       ├── tokenizer.json
│       └── model.safetensors
└── .dl-model-XXXXX/                 # Temp dir during download (cleaned up)
```

---

## Inter-Service Communication

| From | To | Protocol | Path | When |
|------|----|----------|------|------|
| Client | Gateway | HTTP | `POST /generate` | Every inference request |
| Gateway | Engine | HTTP | `POST /inference` | Forwarded from `/generate` |
| Engine | Sidecar | HTTP | `GET /registry/models` | Polling during startup (every 2s) |
| Engine | Sidecar | HTTP | `POST /adapter/fetch/{id}` | On adapter request |
| Sidecar | HuggingFace | HTTPS | `snapshot_download()` | Model/adapter download |
| Sidecar | Redis | TCP | Redis protocol | L2 cache read/write |
| Prometheus | All | HTTP | `GET /metrics` | Scraping (every 15s) |

---

## Configuration Reference

### `.env` File

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:

```bash
# Engine
ENGINE_MODEL_NAME=Qwen/Qwen2-0.5B
ENGINE_MAX_MODEL_LEN=512
ENGINE_GPU_MEM_UTIL=0.70
ENGINE_DTYPE=bfloat16
ENGINE_MAX_PENDING=10
ENABLE_ENGINE_MOCK=false          # Set to true for CPU-only testing

# Sidecar
SIDECAR_SHARED_VOLUME=/mnt/models
SIDECAR_INITIAL_MODEL=Qwen/Qwen2-0.5B
SIDECAR_INITIAL_MODEL_VERSION=main
SIDECAR_MAX_ADAPTERS=10
SIDECAR_L1_CAPACITY_MB=512

# Gateway
GATEWAY_REQUEST_TIMEOUT=300.0

# L2 Cache
L2_REDIS_HOST=localhost
L2_REDIS_PORT=6379

# Auth (required for private models)
HF_TOKEN=hf_your_token_here
```

### docker-compose.yml Override

The `docker-compose.yml` overrides some defaults for local development (e.g., `ENGINE_MODEL_NAME=arnir0/Tiny-LLM` to use a tiny model that downloads quickly).

---

## Testing Guide

### Running the Stack

#### Full stack with GPU (real inference)

```bash
# Build and start all services
docker-compose up -d

# Watch logs
docker-compose logs -f

# Wait for model download + engine initialization
# Watch for "Engine startup complete" in engine logs
docker-compose logs -f engine
```

#### Full stack without GPU (mock engine)

Uncomment `ENABLE_ENGINE_MOCK=true` in `docker-compose.yml` under the engine service, then:

```bash
docker-compose up -d
```

#### Unit tests (no Docker needed)

```bash
make install    # Install dependencies
make test       # Run unit tests
```

### Component-Level Tests

These test each service in isolation. When running the full stack, all services are reachable at `localhost`.

#### Gateway

```bash
# 1. Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# 2. Generate (model exists in routing table)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-8b","prompt":"Hello","max_tokens":50}'
# Expected: 200 with inference response, or 503 if engine is not ready

# 3. Generate (unknown model)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"nonexistent","prompt":"Hello"}'
# Expected: 404 {"detail":"Model nonexistent not found"}
```

#### Engine

```bash
# 1. Health check
curl http://localhost:8080/health
# Expected: {"status":"healthy"} or {"status":"initializing"}

# 2. Readiness check
curl http://localhost:8080/ready
# Expected: {"status":"ready"} or 503

# 3. Inference (basic)
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?","max_tokens":50,"temperature":0.0}'
# Expected: {"text":"...","tokens_generated":N,"duration_seconds":X.XX}

# 4. Inference with LoRA adapter
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"Translate: Hello",
    "max_tokens":50,
    "adapter_identifier":"my-org/my-lora",
    "adapter_version":"latest"
  }'

# 5. Validation - empty prompt (400)
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"","max_tokens":50}'
# Expected: 422 validation error

# 6. Validation - max_tokens out of range (422)
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":99999}'
# Expected: 422 validation error

# 7. Metrics
curl http://localhost:8080/metrics
# Expected: Prometheus text format with engine_requests_total, etc.
```

#### Sidecar

```bash
# 1. Health check
curl http://localhost:8001/health
# Expected: {"status":"ok"}

# 2. Readiness check
curl http://localhost:8001/ready
# Expected: {"status":"ready","resident_models":["arnir0/Tiny-LLM"]}

# 3. Check model registry
curl http://localhost:8001/registry/models
# Expected: {"arnir0/Tiny-LLM":{"model_id":"arnir0/Tiny-LLM","version":"main","status":"loaded","local_path":"..."}}

# 4. Check adapter registry
curl http://localhost:8001/registry/adapters
# Expected: {}

# 5. Load a new model (async - returns 202)
curl -X POST "http://localhost:8001/load/Qwen/Qwen2-0.5B?version=main"
# Expected: 202 {"status":"downloading","model_identifier":"Qwen/Qwen2-0.5B"}

# 6. Poll registry to check download progress
curl http://localhost:8001/registry/models | python3 -m json.tool
# Watch for status change: "downloading" → "loaded"

# 7. Load already-loaded model (idempotent)
curl -X POST "http://localhost:8001/load/arnir0/Tiny-LLM?version=main"
# Expected: 200 {"status":"loaded","model_identifier":"arnir0/Tiny-LLM","local_path":"..."}

# 8. Unload a model
curl -X POST "http://localhost:8001/unload/Qwen/Qwen2-0.5B"
# Expected: {"status":"success","model_identifier":"Qwen/Qwen2-0.5B"}

# 9. Unload non-existent model (404)
curl -X POST "http://localhost:8001/unload/nonexistent/model"
# Expected: 404 {"detail":"Model nonexistent/model not resident."}

# 10. Fetch adapter
curl -X POST "http://localhost:8001/adapter/fetch/my-org/my-adapter?version=latest"
# Expected: 200 {"status":"success","adapter_identifier":"my-org/my-adapter","local_path":"..."}
#    or 500 if adapter doesn't exist on HuggingFace

# 11. Metrics
curl http://localhost:8001/metrics
# Expected: Prometheus text with sidecar_resident_models, etc.
```

### End-to-End Tests

These test the full request path from client to response.

#### Basic inference flow

```bash
# 1. Start the stack
docker-compose up -d

# 2. Wait for readiness (poll until 200)
until curl -sf http://localhost:8080/ready > /dev/null; do
  echo "Waiting for engine..."
  sleep 5
done
echo "Engine ready!"

# 3. Run inference through the gateway
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "prompt": "The meaning of life is",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# 4. Verify metrics were recorded
curl -s http://localhost:8080/metrics | grep engine_requests_total
# Expected: engine_requests_total{model="arnir0/Tiny-LLM",status="success"} 1
```

#### Dynamic model loading

```bash
# 1. Load a new model via sidecar
curl -X POST "http://localhost:8001/load/Qwen/Qwen2-0.5B?version=main"

# 2. Watch download progress
watch -n 2 'curl -s http://localhost:8001/registry/models | python3 -m json.tool'

# 3. Once loaded, verify it appears in the registry
curl -s http://localhost:8001/registry/models | python3 -m json.tool
```

#### Queue pressure test

```bash
# Send multiple concurrent requests to test queue limits (max_pending=10)
for i in $(seq 1 15); do
  curl -s -X POST http://localhost:8080/inference \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Count from 1 to 100","max_tokens":200}' &
done
wait

# Some requests should return 429 (queue full)
```

### Monitoring & Metrics

#### Prometheus queries

```bash
# Access Prometheus at http://localhost:9090

# Request rate by status
rate(engine_requests_total[5m])

# Average request duration
rate(engine_request_duration_seconds_sum[5m]) / rate(engine_request_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95, rate(engine_request_duration_seconds_bucket[5m]))

# Pending requests
engine_pending_requests

# Resident model count
sidecar_resident_models

# Model load duration
histogram_quantile(0.99, rate(sidecar_model_load_duration_seconds_bucket[5m]))
```

#### Quick metrics check

```bash
# All metrics from all services in one go
echo "=== Gateway ===" && curl -s http://localhost:8000/metrics 2>/dev/null || echo "(no metrics endpoint)"
echo "=== Engine ===" && curl -s http://localhost:8080/metrics | head -20
echo "=== Sidecar ===" && curl -s http://localhost:8001/metrics | head -20
```

---

## Shared Modules

### `shared/types.py`

Shared data types used across components:

| Type | Fields | Description |
|------|--------|-------------|
| `BlockReference` | `device_id`, `memory_address`, `size_bytes` | Reference to a KV cache block in GPU HBM |
| `TransferResult` | `success`, `message`, `data` | Result of a cache transfer operation |
| `StorageNodeInfo` | `node_id`, `host`, `port`, `health` | L2 storage node descriptor |
| `AllocationPointer` | `cpu_address`, `size_bytes` | CPU DRAM allocation |

### `shared/ports.py`

Protocol interfaces for dependency injection and testing:

| Protocol | Methods | Used By |
|----------|---------|---------|
| `CacheStore` | `put(key, data)`, `get(key)` | L1 and L2 cache implementations |
| `ModelRepository` | `fetch(identifier, version)` | Artifact fetching abstraction |
| `GPUTransfer` | `copy_hbm_to_dram()`, `copy_dram_to_hbm()` | CUDA memory transfers |

---

## Project Commands (Makefile)

```bash
make install     # Install dependencies (uv sync --extra dev)
make test        # Run unit tests
make test-int    # Run integration tests
make lint        # Lint with ruff
make format      # Format with ruff
make typecheck   # Type check with mypy
make docker-up   # docker-compose up -d
make docker-down # docker-compose down
make clean       # Remove cache files
```
