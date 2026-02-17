# LLM Inference Server — Roadmap

## Requirements

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Accept HTTP POST with `{model_id, prompt, max_tokens, temperature, stream}`, return text or SSE stream |
| FR-2 | Load/unload/hot-swap LLM models at runtime via sidecar without restarting engine |
| FR-3 | Apply/unapply LoRA adapters on a resident base model; multiple adapters share base weights |
| FR-4 | L1 (CPU DRAM) and L2 (Redis) KV cache with automatic tiered fallback |
| FR-5 | Gateway routes requests to pod maximizing cache reuse (prefix hash affinity) |
| FR-6 | Model/adapter registry with persistence across restarts |
| FR-7 | SSE streaming of tokens as generated, with client disconnect cancellation |
| FR-8 | `/health` and `/ready` probes on every service |
| FR-9 | Queue limits (429), request timeouts (504), graceful shutdown |
| FR-10 | Prometheus `/metrics` endpoint on every service |

### Non-Functional Requirements

| NFR | Value | Notes |
|-----|-------|-------|
| Target dev model | `Qwen/Qwen2-0.5B` | ~1GB VRAM, fast iteration |
| GPU | NVIDIA 8GB (RTX 3060/4060) | 8GB VRAM budget |
| Max concurrent requests | 10 | Starting point |
| Max context length | 512 (dev) | Keep low for fast tests |
| GPU memory utilization | 0.70 | Leave ~2.4GB for OS/desktop |
| L1 cache budget | 512 MB | CPU DRAM for KV cache |
| L2 cache budget | 256 MB | Redis maxmemory |
| Adapter limit | 10 | Max resident adapters |
| Model artifact source | HuggingFace Hub | `huggingface_hub.snapshot_download()` |

### SLOs

| Metric | Target |
|--------|--------|
| Time to first token (TTFT) | < 500 ms |
| Token generation latency | < 50 ms/token |
| GPU memory alert threshold | 90% of 8GB budget |

---

## Implementation Roadmap

### Phase A — Project Foundation

Make the codebase importable and establish architectural conventions. Create all missing `__init__.py` files, add dependencies to `pyproject.toml`, set up `shared/` package with common types (`BlockReference`, `StorageNodeInfo`, `TransferResult`) and Protocol interfaces (`CacheStore`, `ModelRepository`, `GPUTransfer`). Scrub secrets and set up `.env.example`.

### Phase B — Configuration & Test Infrastructure

Add `pydantic-settings` config modules for each service (engine, sidecar, gateway). Set up pytest infrastructure with shared fixtures, fakes, and a first smoke test. Create a Makefile for common operations (install, test, lint, format, proto, docker-up).

### Phase C — Engine (Core Inference Path)

Fix the vLLM engine wrapper: correct async bugs, replace hardcoded values with config, fix API imports and request models. Add `/health` and `/ready` endpoints, Prometheus metrics, structured error handling (422, 500), and an integration test with a real model on GPU.

### Phase D — Sidecar (Model Management)

Fix broken imports and config injection in the sidecar. Implement real model downloads via HuggingFace Hub, add Prometheus metrics, write unit tests for all sidecar API routes, and build the sidecar Dockerfile.

### Phase E — Docker Infrastructure

Write Dockerfiles for all four services (gateway, engine, sidecar, controller). Create `docker-compose.yml` orchestrating gateway, engine, sidecar, Redis, and Prometheus with shared volumes and GPU passthrough.

### Phase F — Token Streaming

Implement queue-based streaming in the engine batching loop. Add an SSE endpoint (`text/event-stream`) for `stream=true` requests. Handle client disconnect with generation abort and queue cleanup. Add streaming-specific metrics (tokens emitted, stream duration, cancellations).

### Phase G — LoRA Adapter Support

Fix engine LoRA loading (remove hardcoded paths, add `--enable-lora` flag). Fix sidecar adapter response contract for path consistency. Add adapter tracking to prevent duplicate loads and pass `lora_request` to vLLM. Add LoRA metrics and an integration test.

### Phase H — Registry Persistence

Add JSON file persistence to the model/adapter registry so state survives restarts. Add detailed query endpoints (`/registry/models`, `/registry/adapters`). Extend registry metadata with tags, warmup prompts, and preferred device.

### Phase I — KV Cache (L1 Only)

Fix L1 cache import paths. Implement `check_global_availability()` and `execute_l1_eviction()` in the cache manager. Add L1 cache metrics (capacity, used, hits, misses, evictions). Write unit tests for the allocator, LRU eviction, put/get, and multi-tier offload. **Note:** this phase covers the local CPU DRAM cache only — the distributed L2 cache (Redis cluster with controller/storage nodes) remains scaffolding and is not made runnable in this roadmap.

### Phase J — Cache-Aware Routing

Fix gateway routing imports and config. Add dynamic backend registration/deregistration endpoints. Build a Router class with cache-hit scoring and least-connections fallback. Add prefix hash computation for cache affinity. Add gateway metrics and unit tests.

### Phase K — Robustness

Add request queue limits (HTTP 429 with Retry-After). Add request timeouts (HTTP 504 with vLLM abort). Handle client disconnect for non-streaming requests. Implement graceful shutdown (drain in-flight, stop accepting, cleanup). Write a robustness test suite covering all cases.

### Phase L — Sidecar Completion

Add download retries (3x exponential backoff) with atomic downloads (tmp + rename). Add model warmup (send prompts to engine after load). Create `kv_cache.proto` and generate gRPC stubs. Update KV cache and L2 cache handler imports to use real proto stubs (interface wiring only — the distributed cache backend with controller/storage nodes is **not** implemented in this roadmap). Add structured JSON logging. Write a full lifecycle integration test.

### Phase M — Batching Metric

Add batch size and step latency metrics to the engine batching loop. Verify with concurrent requests.

### Out of Scope (Future Work)

- **Distributed KV cache (L2)** — the controller (`KVCacheWatcher`), storage nodes, consistent hashing ring, cross-node transfers, and health monitoring are scaffolded but not made runnable. Only the `.proto` definition and gRPC stub wiring are covered.
- **Multi-node deployment** — Kubernetes manifests, Helm charts, horizontal scaling.
- **InfiniStore** or alternative cache backends.

---

## Execution Order

```
Sprint 1:  Phase A + B            —  Foundation & testability
Sprint 2:  Phase C + D (parallel) —  Core engine + sidecar
Sprint 3:  Phase E + I            —  Docker infra + L1 cache
Sprint 4:  Phase F + G + H        —  Streaming, LoRA, registry (parallel)
Sprint 5:  Phase J + K + L + M    —  Routing, robustness, completion
```

### Dependency Graph

```
A ──► B ──► C ──► F (streaming)
             │    ├► G (LoRA) ──────────────────┐
             │    ├► K (robustness)              │
             │    └► M (batching metric)         │
             │                                   ▼
             ├► D ──► H (registry) ──────────► E (Docker)
             │    └► L (sidecar completion)
             │
             └► I (L1 cache) ──► J (routing)
```

## Verification Checklist

When all phases are complete, the following should work end-to-end:

1. `make install` — installs all deps
2. `make lint` — no lint errors
3. `make test` — all unit tests pass (~30+ tests)
4. `docker-compose up -d` — all services start
5. `curl localhost:8000/health` — gateway healthy
6. `curl localhost:8080/ready` — engine ready with model loaded
7. `curl -X POST localhost:8000/generate -d '{"model":"Qwen/Qwen2-0.5B","prompt":"Hello","max_tokens":10}'` — returns generated text
8. `curl localhost:8080/metrics` — Prometheus metrics present
9. `curl localhost:9090/targets` — Prometheus scraping all services
10. `make test-int` — integration tests pass
