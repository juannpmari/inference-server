# Backlog: Architecture Analysis & gRPC Migration

## Current Architecture Overview

```
                    +-------------+
                    |   Clients   |
                    +------+------+
                           | HTTP POST /generate
                    +------v------+
                    |   Gateway   | :8000
                    |  (routing)  |
                    +------+------+
                           | HTTP POST /inference  (static MODEL_SERVICE_MAP)
              +------------+------------+
              |            |            |
        +-----v-----++----v----++-----v-----+
        |  Pod A    ||  Pod B  ||  Pod N    |
        |+---------+||         ||           |
        || Engine  |||  ...    ||   ...     |
        ||  :8080  |||         ||           |
        |+---------+||         ||           |
        ||HTTP+gRPC|||         ||           |
        |+---------+||         ||           |
        || Sidecar |||         ||           |
        ||:8001    |||         ||           |
        ||:50051   |||         ||           |
        |+---------+||         ||           |
        +-----------++---------++-----------+
```

### Communication Paths

| Path | Protocol | Pattern | Latency-sensitive? |
|---|---|---|---|
| Gateway -> Engine | HTTP/REST | Synchronous request/response (300s timeout) | Yes (user-facing) |
| Engine -> Sidecar (model/adapter mgmt) | HTTP/REST | Fire-and-forget + polling | No (startup/cold-load) |
| Engine <-> Sidecar (KV cache) | gRPC | High-frequency block store/load (5s timeout, 16MB max msg) | Yes (hot path) |

### Communication Matrix

```
+----------+---------+----------------------------------------------+
| Source   | Target  | Protocol / Endpoint                          |
+----------+---------+----------------------------------------------+
| Client   | Gateway | HTTP POST /generate (timeout: 300s)          |
| Gateway  | Engine  | HTTP POST /inference (timeout: 300s)         |
| Engine   | Sidecar | HTTP GET /registry/models (polling, 2s)      |
| Engine   | Sidecar | HTTP POST /load/{model} (fire-and-forget)    |
| Engine   | Sidecar | HTTP POST /adapter/load (fire-and-forget)    |
| Engine   | Sidecar | HTTP GET /registry/adapters (polling, 1s)    |
| Engine   | Sidecar | gRPC StoreBlock/LoadBlock (5s per RPC)       |
| Sidecar  | HF Hub  | HTTPS (huggingface_hub library)              |
| Sidecar  | Redis   | async Redis (L2 cache, consistent hash)      |
+----------+---------+----------------------------------------------+
```

---

## REST API Surface

### Gateway (Port 8000)

File: `data_plane/gateway/routing.py`

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/health` | GET | Liveness probe | - | `{"status": "healthy"}` |
| `/generate` | POST | Route inference requests | `{"model": str, "prompt": str, ...}` | JSON response or 404/503 |

Routing logic:
- Uses static `MODEL_SERVICE_MAP` mapping model names to engine URLs.
- Maps model "llama-3-8b" to "http://engine:8080".
- Forwards requests to `/inference` endpoint on target engine.
- Client timeout: 300 seconds (5 minutes).
- Error handling: connect errors return 503, other errors return 500.

### Engine (Port 8080)

File: `data_plane/inference/engine/api.py`

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/health` | GET | Liveness probe | - | `{"status": "healthy"}` or 503 if initializing |
| `/ready` | GET | Readiness probe | - | `{"status": "ready"}` or 503 if model not loaded |
| `/inference` | POST | Generate text | `InferenceRequest` | `InferenceResponse` or error |
| `/metrics` | GET | Prometheus metrics | - | Prometheus format |

InferenceRequest model:
- `prompt`: str (required, min 1 char)
- `max_tokens`: int (default 256, range 1-4096)
- `temperature`: float (default 0.7, range 0-2.0)
- `stream`: bool (default False, stubbed for future SSE support)
- `adapter_identifier`: Optional[str] (LoRA adapter ID)
- `adapter_version`: Optional[str] (LoRA adapter version)

InferenceResponse model:
- `text`: str
- `tokens_generated`: int
- `duration_seconds`: float

Engine startup flow:
1. Backend in `lifespan()` spawns `_init_engine()` as background task.
2. Polls sidecar at `{sidecar_url}/registry/models` every 2s (configurable).
3. Waits up to 600s (configurable) for model `status="loaded"`.
4. Creates Engine/MockEngine with resolved model path.
5. Starts `continuous_batching_loop()` async task.
6. HTTP server accepts requests immediately (even while engine still initializing).

Queue management:
- Max pending requests: 10 (configurable via `ENGINE_MAX_PENDING`).
- Request timeout: 300 seconds.
- Returns 429 (Too Many Requests) when queue full.

### Sidecar (Port 8001)

File: `data_plane/inference/sidecar/api.py`

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/health` | GET | Liveness probe | - | `{"status": "ok"}` or 503 |
| `/ready` | GET | Readiness probe | - | `{"status": "ready", "resident_models": [...]}` or 503 |
| `/load/{model_identifier}` | POST | Load model (async) | `version: str` (query) | 202 Accepted or 200 (already loaded) |
| `/unload/{model_identifier}` | POST | Remove model | - | `{"status": "success"}` or 404 |
| `/registry/models` | GET | List resident models | - | Dict of models with metadata |
| `/adapter/load/{adapter_id}` | POST | Load LoRA adapter (async) | `version: str` (query) | 202 Accepted or 200 |
| `/adapter/unload/{adapter_id}` | POST | Remove adapter | - | `{"status": "success"}` or 404 |
| `/registry/adapters` | GET | List resident adapters | - | Dict of adapters |
| `/cache/blocks` | GET | Query cached KV blocks | `prefix_hash?: str`, `model_id?: str` | Array of block metadata |
| `/cache/stats` | GET | Cache statistics | - | Cache stats dict |
| `/metrics` | GET | Prometheus metrics | - | Prometheus format |

Model registry entry shape:
```json
{
    "model_id": "str",
    "version": "str",
    "status": "downloading | loaded",
    "local_path": "str (when loaded)",
    "tags": ["optional"],
    "warmup_prompts": ["optional"]
}
```

Fire-and-forget pattern:
- Model/adapter load returns 202 Accepted immediately.
- Backend task downloads in background.
- Client polls `/registry/models` or `/registry/adapters` to check status.
- Status transitions: "downloading" -> "loaded".

### gRPC Service (Port 50051)

File: `shared/proto/kv_cache.proto`

Service: `kv_cache.KVCacheService`

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| StoreBlock | block_id, block_hash, data (bytes), model_id, layer_name | success, message | Store KV bytes in sidecar L1 |
| LoadBlock | block_id, layer_name | success, data (bytes), message | Retrieve KV bytes from sidecar |
| GetFreeBlocks | (empty) | num_free_blocks | Query available L1 slots |
| AllocateBlocks | block_hashes (repeated string) | success, block_ids (repeated int32), message | Pre-allocate block slots |
| FreeBlock | block_id | success, message | Release block slot |

gRPC configuration:
- Channel: `grpc.aio.insecure_channel()`
- Max message size: 16 MB (both send and receive)
- RPC timeout: 5 seconds per call

---

## Model Loading Flow

### Full Request Path: Gateway -> Engine -> Sidecar

```
1. CLIENT SENDS REQUEST TO GATEWAY
   POST /generate {"model": "llama-3-8b", "prompt": "..."}
                          |
                          v
2. GATEWAY ROUTES TO ENGINE
   httpx.post("http://engine:8080/inference", ...)
   timeout: 300s
                          |
                          v
3. ENGINE STARTUP (Background, happens once)
   Triggered by app lifespan() -> _init_engine()

   a) Poll sidecar for model:
      GET /registry/models every 2s
      Timeout: 600s
      Check: entry["llama-3-8b"]["status"] == "loaded"

   b) While engine is initializing:
      /health -> 503 (initializing)
      /ready  -> 503 (model not ready yet)

   c) Once resolved_model_path obtained:
      Engine.init(model_path) or MockEngine.init()
      Start continuous_batching_loop()
      /ready -> 200 (ready for requests)
                          |
                          v
4. SIDECAR MODEL LOADING (Background, happens in parallel)
   Triggered by sidecar startup

   a) Sidecar lifespan calls _initial_load() background task
      POST /load/{initial_model}

   b) Background fetch from HuggingFace:
      Uses huggingface_hub.snapshot_download()
      Atomic download: temp dir -> rename
      Path: /mnt/models/{model_id}/{version}/

   c) Status updates:
      "downloading" -> (fetch completes) -> "loaded"

   d) Persistence:
      Writes registry.json to disk
      Restores on sidecar restart
                          |
                          v
5. ENGINE INFERENCE
   POST /inference with InferenceRequest

   a) Add request to engine batch queue
   b) continuous_batching_loop():
      - Polls engine.step() in thread
      - Collects outputs
      - Resolves request futures
   c) Return InferenceResponse
      - text, tokens_generated, duration_seconds
```

### LoRA Adapter Loading (Optional)

```
InferenceRequest with adapter_identifier="peft/lora-1", adapter_version="v1"
  |
  v
Engine.add_request() -> LoRAManager.ensure_adapter_loaded()
  |
  v
Check if already loaded on GPU (fast path) -> Return
  |
  v (cache miss)
Trigger download:
  POST http://sidecar:8001/adapter/load/peft/lora-1?version=v1
  |
  v
Response: 202 (downloading) or 200 (cached)
  |
  v
Poll /registry/adapters until entry status == "loaded"
  Polling interval: 1s, timeout: 600s
  |
  v
Evict LRU adapter if at capacity (max_loras=4 by default)
  Call engine.remove_lora(evicted_int_id)
  |
  v
Load adapter onto GPU:
  lora_request = LoRARequest(lora_name, lora_int_id, local_path)
  engine.add_lora(lora_request)  # via asyncio.to_thread()
  |
  v
Add to LoRAManager._loaded OrderedDict (LRU tracking)
Pass lora_request to engine.add_request()
```

---

## HTTP Client Configuration

### Gateway Client (httpx)

File: `data_plane/gateway/routing.py`

```python
httpx.AsyncClient(timeout=300.0)
```

- Used for forwarding requests to engine.
- Connection pooling: yes (stored in app.state, one client per app lifespan).
- Retry strategy: none (relies on Kubernetes restart).
- Error handling: ConnectError -> 503, other -> 500.

### Engine -> Sidecar Clients (httpx)

File: `data_plane/inference/engine/lora_manager.py`

```python
httpx.AsyncClient(timeout=30.0)
```

- Used for triggering adapter loads and polling registry.
- Per-call timeout: 5 seconds.
- Retry: none (polling loop handles transient failures).

File: `data_plane/inference/engine/api.py`

```python
httpx.AsyncClient()  # Used in _wait_for_sidecar_model()
```

- Per-call timeout: 5 seconds.
- Polling: 2s interval, 600s deadline.

### Connection Pooling & Health Checks

- Gateway: single AsyncClient per app (lifespan manages creation/closing).
- Sidecar polling: new AsyncClient per poll attempt (not pooled).
- Health checks: implicit via `/health` and `/ready` endpoints.
- Service discovery: static MODEL_SERVICE_MAP (Kubernetes DNS via service names).

---

## Key Configuration Values

| Component | Setting | Value | File |
|-----------|---------|-------|------|
| Gateway | Host | 0.0.0.0 | gateway/config.py |
| Gateway | Port | 8000 | gateway/config.py |
| Gateway | Request timeout | 300s | gateway/routing.py |
| Engine | Host | 0.0.0.0 | engine/config.py |
| Engine | Port | 8080 | engine/config.py |
| Engine | Model name | "Qwen/Qwen2-0.5B" | engine/config.py |
| Engine | Sidecar URL | "http://localhost:8001" | engine/config.py |
| Engine | Sidecar poll interval | 2.0s | engine/config.py |
| Engine | Sidecar timeout | 600s | engine/config.py |
| Engine | Max pending requests | 10 | engine/config.py |
| Engine | Adapter poll interval | 1.0s | engine/config.py |
| Engine | Adapter poll timeout | 600s | engine/config.py |
| Engine | KV offload enabled | False | engine/config.py |
| Engine | KV offload blocks | 1024 | engine/config.py |
| Engine | gRPC URL | "localhost:50051" | engine/config.py |
| Engine | LoRA enabled | False | engine/config.py |
| Engine | Max LoRAs | 4 | engine/config.py |
| Sidecar | Host | 0.0.0.0 | sidecar/config.py |
| Sidecar | Port | 8001 | sidecar/config.py |
| Sidecar | gRPC port | 50051 | sidecar/config.py |
| Sidecar | Model store | "/mnt/models" | sidecar/config.py |
| Sidecar | L1 capacity | 512 MB | sidecar/config.py |
| Sidecar | L1 blocks | 1024 | sidecar/config.py |
| Sidecar | L1 block size | 128 KB | sidecar/config.py |
| Sidecar | Max adapters | 10 | sidecar/config.py |
| gRPC | Max message size | 16 MB | sidecar/kv_cache_api.py |
| gRPC | RPC timeout | 5s | engine/sidecar_cache_client.py |

---

## Analysis: gRPC Migration

### Path 1 - Gateway -> Engine: RECOMMENDED - Switch to gRPC

This is where gRPC gives the most architectural leverage at scale.

**Multiplexing (HTTP/2):** With many pods behind the gateway, HTTP/1.1 means one request per TCP connection (or head-of-line blocking with connection reuse). HTTP/2 multiplexes streams over a single connection per pod -- critical when 1 gateway fans out to N engine pods.

**Streaming for token generation:** The `stream: bool` field is already stubbed in `InferenceRequest`. With gRPC server-streaming, you get first-class token-by-token streaming with backpressure and flow control built in. With REST you'd need SSE (half-duplex, no backpressure) or WebSockets (no schema).

**Connection management at scale:** `httpx.AsyncClient` with a 300s timeout to dozens of pods means the gateway holds many long-lived HTTP/1.1 connections. gRPC channels handle this more efficiently with persistent HTTP/2 connections, automatic reconnection, and built-in keepalives.

**Typed contracts:** A `.proto` for the inference API gives a compile-time contract between gateway and engine. Currently the gateway just forwards a raw `dict` -- no validation until it hits the engine.

**Load balancing awareness:** gRPC integrates naturally with client-side load balancing (round-robin, weighted, pick-first) via name resolution. This replaces the static `MODEL_SERVICE_MAP` with something dynamic. You could use `dns:///engine-headless:8080` to resolve all pod IPs behind a headless K8s Service.

**Built-in deadlines and cancellation:** gRPC propagates deadlines across the call chain. If the client disconnects, the gateway can cancel the in-flight RPC to the engine. The current REST chain has no cancellation propagation.

### Path 2 - Engine -> Sidecar (model/adapter management): NOT RECOMMENDED - Keep REST

This is a low-frequency, intra-pod communication path. The current pattern is:

```
POST /load/{model}  ->  202 Accepted
Poll GET /registry/models every 2s until status="loaded"
```

gRPC would let you replace this with a server-streaming RPC (subscribe to load status), but:

- These calls happen at startup or on cold adapter loads -- not on the hot path.
- Engine and sidecar are colocated in the same pod (localhost), so latency is negligible.
- The polling pattern is simple and works fine.
- Adding a second `.proto` + service definition for a handful of management RPCs adds complexity with little benefit.

### Path 3 - Engine <-> Sidecar (KV cache): ALREADY gRPC - No change needed

KV block transfers are latency-sensitive, high-frequency, and involve large binary payloads (up to 16MB). gRPC is the right choice here. Already implemented correctly.

### Summary Table

| Path | Current | Recommendation | Priority |
|---|---|---|---|
| Gateway -> Engine | REST/HTTP | **Switch to gRPC** (streaming, multiplexing, LB, deadlines) | **High** -- biggest scalability win |
| Engine -> Sidecar (mgmt) | REST/HTTP | **Keep REST** (low-frequency, intra-pod, simple) | N/A |
| Engine <-> Sidecar (KV) | gRPC | **Keep gRPC** (already correct) | N/A |

---

## Scalability Recommendations for Multi-Pod Architecture

### A. Replace MODEL_SERVICE_MAP with gRPC service discovery

```
Current:  MODEL_SERVICE_MAP = {"llama-3-8b": "http://engine:8080"}
Proposed: gRPC channel to headless K8s Service with client-side LB
```

A Kubernetes headless Service (`.spec.clusterIP: None`) returns all pod IPs in DNS. gRPC's built-in DNS resolver + round-robin policy distributes requests across pods automatically. No static map needed.

### B. Server-streaming for token generation

```protobuf
service InferenceService {
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
}
```

This replaces the future SSE approach with a proper streaming protocol. Each token chunk is a `GenerateResponse` message. The gateway can forward the stream to the HTTP client as SSE or chunked responses.

### C. Health checking via gRPC Health protocol

gRPC has a standard health checking protocol (`grpc.health.v1.Health`). Benefits:
- The gateway can track which engine pods are healthy without a separate HTTP health-check loop.
- Kubernetes can use `grpc` liveness/readiness probes (supported since K8s 1.24).
- Load balancing decisions incorporate health status automatically.

### D. Deadline propagation

With gRPC, the gateway sets a deadline (e.g., 30s), and if the engine can't finish in time, the RPC is cancelled end-to-end. Currently, the 300s httpx timeout on the gateway side has no way to tell the engine "stop working, the client is gone."

---

## Streaming Endpoints (Current State)

No streaming endpoints are currently implemented. Infrastructure is partially prepared:

- `InferenceRequest.stream: bool = False` -- flag exists but unused.
- `sse-starlette` dependency listed in `pyproject.toml`.
- FastAPI supports `StreamingResponse` natively.

What's needed for streaming:
1. Change Engine inference to yield tokens incrementally.
2. Return `StreamingResponse` with `stream_tokens()` generator (if staying REST) or implement server-streaming RPC (if moving to gRPC).
3. Update batching loop to handle token-level outputs.

---

## Health Checks & Probes

| Service | Endpoint | Indicates | Status Code |
|---------|----------|-----------|-------------|
| Engine | `/health` | Process alive | 200/503 |
| Engine | `/ready` | Ready for requests | 200/503 |
| Sidecar | `/health` | Process alive | 200/503 |
| Sidecar | `/ready` | Models loaded | 200/503 |
| Gateway | `/health` | Process alive | 200 |

No active health-check loop in the gateway. Relies on Kubernetes to restart failed pods. Passive error handling: ConnectError returns 503, other errors return 500.
