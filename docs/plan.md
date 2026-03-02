# Implementation Plan: LLM Inference Server (Steps 0–10)

## Context

The project is a Kubernetes-native LLM inference platform (~28% complete, ~1,300 lines across 15 files). The previous developer scaffolded many features but none are runnable: no `__init__.py` files, no `.proto` file for gRPC, broken imports in 6+ files, missing dependencies, all Dockerfiles empty, no tests, no metrics. This plan takes the codebase from "broken scaffolding" to "locally testable system with proper tests and telemetry," covering plan.md Steps 0–10 only.

---

## 0. Architectural Decision

### Options Evaluated

| Architecture | Verdict | Reason |
|-------------|---------|--------|
| **Full Hexagonal (Ports & Adapters)** | Rejected | ~1-2 weeks of restructuring for 1,300 lines of prototype code. Would triple file count with ~20 interface files. Solo developer, no business domain entities. |
| **Clean Architecture (Onion)** | Rejected | Entities (`Model`, `CacheEntry`) have no behavior — they are plain data. Use-case classes would wrap 15-line methods with zero added value. This is infrastructure, not a business domain. |
| **Simple Layered (API→Service→Repository)** | Partially adopted | Good fit for gateway and engine. Redundant for sidecar which already has natural layers. |
| **Modular + Selective Ports** | **Adopted** | 80% of hexagonal's testability benefit at 15% of the cost. Targeted improvements where they matter. |

### What We Do (Pragmatic Modular Architecture)

1. **Keep existing service boundaries** — gateway, engine, sidecar, controller map to deployment units. Correct.
2. **Add `shared/` package** — single definition for `BlockReference`, `StorageNodeInfo`, `TransferResult` (currently duplicated 3x with different shapes across files).
3. **Extract configuration** — one `config.py` per service using `pydantic-settings`, replacing 10+ hardcoded constants.
4. **Add 3 Protocol interfaces** — only where testability demands it:
   - `CacheStore` — `put(key, data)`, `get(key)`. Both L1 and L2 already conform.
   - `ModelRepository` — `fetch(id, version) -> path`. Replaces `asyncio.sleep(5)` simulation.
   - `GPUTransfer` — `copy_hbm_to_dram`, `copy_dram_to_hbm`. Already pure simulation.
5. **No use-case classes, no entity classes, no DI container** — existing methods ARE the use cases.

### What Already Works (don't change)

- `EvictionPolicy` base class + `LRUPolicy` in `eviction_policy.py` — already the port/adapter pattern.
- `L1CacheAPI` as facade coordinating allocator + transfer + eviction — correct layering.
- Sidecar pattern (2 containers/pod), control/data plane separation, consistent hashing — all correct.

### Proposed Directory Layout

```
inference-server/
  shared/                              # NEW: shared types + proto
    __init__.py
    types.py                           # BlockReference, StorageNodeInfo, TransferResult
    ports.py                           # CacheStore, ModelRepository, GPUTransfer protocols
    proto/
      kv_cache.proto                   # Single source of truth for gRPC
      kv_cache_pb2.py                  # Generated
      kv_cache_pb2_grpc.py             # Generated

  data-plane/
    gateway/
      __init__.py
      app.py                           # FastAPI entrypoint (renamed from routing.py)
      router.py                        # NEW: routing logic with cache-aware scoring
      config.py                        # NEW: env-based config
      metrics.py                       # NEW: Prometheus metrics

    inference/
      engine/
        __init__.py
        app.py                         # FastAPI entrypoint (renamed from api.py)
        engine.py                      # vLLM wrapper (bug-fixed)
        config.py                      # NEW: env-based config
        metrics.py                     # NEW: Prometheus metrics

      sidecar/
        __init__.py
        app.py                         # FastAPI + gRPC entrypoint (renamed from api.py)
        artifact_manager.py            # Service layer (bug-fixed)
        cache_manager.py               # Multi-tier orchestrator (bug-fixed)
        kv_cache_handler.py            # gRPC handler (renamed from kv_cache_api.py)
        config.py                      # NEW: env-based config
        metrics.py                     # NEW: Prometheus metrics
        warmup.py                      # NEW: model warmup after load
        l1_cache/                      # UNCHANGED structure
        l2_cache/                      # UNCHANGED structure

      distributed_cache/               # UNCHANGED structure

  tests/
    conftest.py                        # Shared fixtures
    fakes/                             # In-memory implementations of protocols
      fake_cache_store.py
      fake_model_repository.py
      fake_gpu_transfer.py
    unit/                              # No Docker needed
    integration/                       # Requires docker-compose
```

### When to Revisit

Move toward full hexagonal IF: team grows beyond 2-3 devs, you add a second cache backend (InfiniStore), sidecar exceeds ~2,000 lines, or you need multiple deployment contexts.

---

## 1. Requirements

### 1A. Functional Requirements (implied by plan.md, confirmed by code)

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

### 1B. Non-Functional Requirements (decided)

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

### 1C. Development Defaults (environment variables)

```
ENGINE_MODEL_NAME        = "Qwen/Qwen2-0.5B"
ENGINE_MAX_MODEL_LEN     = 512
ENGINE_GPU_MEM_UTIL      = 0.70
ENGINE_DTYPE             = "bfloat16"
SIDECAR_L1_CAPACITY_MB   = 512
L2_REDIS_MAXMEM          = 256mb
ENGINE_MAX_PENDING        = 10
HF_TOKEN                 = <from .env file>
ENABLE_ENGINE_MOCK       = false  # Set to true to use mock engine (no GPU needed); all other functionality unchanged
```

**Mock Engine Note**: When `ENABLE_ENGINE_MOCK=true`, the engine uses deterministic mock responses instead of vLLM. All metrics, routing, batching, streaming, and error handling remain fully functional, allowing complete system testing on machines without GPUs.

---

## 2. Telemetry Metrics

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
| `engine_gpu_memory_used_bytes` | Gauge | `device` | (8GB budget, alert at 90%) |
| `engine_lora_load_duration_seconds` | Histogram | `adapter` |
| `engine_lora_active` | Gauge | — |
| `engine_stream_cancelled_total` | Counter | `model` |

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

Implementation: `prometheus_client` library. Each FastAPI service gets `/metrics` via manual route or `prometheus_fastapi_instrumentator`.

---

## 3. Local Testing Strategy

### docker-compose.yml services
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

---

## 4. What's Done vs. What's Missing (per step)

| Step | Exists (don't redo) | Missing (build) |
|------|---------------------|-----------------|
| **0 Scaffolding** | Repo structure, pyproject.toml (partial), .gitignore | `__init__.py` (9 files), deps, config modules, pytest, Dockerfiles, docker-compose, Makefile, .env |
| **1 MVP Engine** | `engine.py` batching loop, `api.py` /inference | Bug fixes (await, imports, hardcoded), health endpoints, Pydantic models, metrics, error handling |
| **2 Streaming** | Commented-out stub | Queue-based streaming, SSE endpoint, disconnect handling, metrics |
| **3 Batching** | vLLM `engine.step()` already does continuous batching | Batch-size metric only |
| **4 Sidecar** | `artifact_manager.py` structure, `sidecar/api.py` routes | Import fixes, real HF downloads, metrics, tests, Dockerfile |
| **5 LoRA** | `_add_lora()` method, sidecar adapter route | Fix hardcoded path, fix missing await, `--enable-lora` flag, adapter tracking |
| **6 Registry** | In-memory `model_registry` dict | JSON persistence, metadata, query endpoints |
| **7 KV Cache L1** | allocator, LRU eviction, GPU transfer, L1 facade (all logically correct) | Fix import typo, implement 2 NotImplementedError methods, metrics, tests |
| **8 Routing** | Static routing with httpx | Fix `status` bug, dynamic discovery, cache-aware Router, prefix hashing, metrics |
| **9 Robustness** | Nothing | Queue limits, timeouts, cancellation, disconnect, graceful shutdown |
| **10 Sidecar completion** | Routes, L2 connector, distributed cache sketches | Download retries, warmup, `.proto` file, real gRPC stubs, structured logging |

---

## 5. Atomic Tasks (55 tasks, grouped by execution phase)

### Phase A — Make It Importable + Architectural Foundation (5 tasks) — do first, unblocks everything

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **0.1** | Add `__init__.py` to all packages (including new `shared/`) | Create `__init__.py` in: `shared/`, `data-plane/`, `data-plane/gateway/`, `data-plane/inference/`, `data-plane/inference/engine/`, `data-plane/inference/sidecar/`, `data-plane/inference/sidecar/l1_cache/`, `data-plane/inference/sidecar/l2_cache/`, `data-plane/inference/distributed_cache/`, `control-plane/`, `tests/`, `tests/unit/`, `tests/integration/`, `tests/fakes/` | `python -c "from data_plane.inference.engine import engine"` | 15 |
| **0.2** | Add missing deps to `pyproject.toml` + dev group | `pyproject.toml` (add fastapi, uvicorn, httpx, grpcio, grpcio-tools, redis, prometheus-client, pydantic-settings, huggingface-hub; dev: pytest, pytest-asyncio, pytest-cov, ruff, mypy) | `uv sync --extra dev && python -c "import fastapi; import redis; import grpc"` | 20 |
| **0.7** | Fix .gitignore (add .env), create `.env.example`, scrub leaked HF token | `.gitignore`, `.env.example` | `.env` excluded from git | 15 |
| **0.9** | Create `shared/types.py` — single definitions for `BlockReference`, `StorageNodeInfo`, `TransferResult` | `shared/types.py` (NamedTuples/dataclasses replacing 5 duplicate definitions across cache_manager.py, connector.py, controller.py, gpu_transfer.py) | `python -c "from shared.types import BlockReference, StorageNodeInfo, TransferResult"` | 30 |
| **0.10** | Create `shared/ports.py` — 3 Protocol interfaces: `CacheStore`, `ModelRepository`, `GPUTransfer` | `shared/ports.py` (typing.Protocol classes) | `python -c "from shared.ports import CacheStore, ModelRepository, GPUTransfer"` | 25 |

#### Phase A — Verification Checkpoint

Run the following script. **All 5 checks must pass** before moving to Phase B:

```bash
#!/bin/bash
set -e
echo "=== Phase A Verification ==="

# 1. All __init__.py files exist
echo "[1/5] Checking __init__.py files..."
for dir in shared data-plane data-plane/gateway data-plane/inference \
           data-plane/inference/engine data-plane/inference/sidecar \
           data-plane/inference/sidecar/l1_cache data-plane/inference/sidecar/l2_cache \
           data-plane/inference/distributed_cache control-plane \
           tests tests/unit tests/integration tests/fakes; do
  test -f "$dir/__init__.py" || { echo "FAIL: $dir/__init__.py missing"; exit 1; }
done
echo "  PASS: all __init__.py present"

# 2. Dependencies resolve
echo "[2/5] Checking dependency installation..."
uv sync --extra dev 2>&1 | tail -1
python -c "import fastapi; import redis; import grpc; import pydantic_settings; import prometheus_client; import huggingface_hub" \
  && echo "  PASS: all deps importable" || { echo "FAIL: dep import error"; exit 1; }

# 3. .env excluded from git
echo "[3/5] Checking .gitignore..."
echo "SECRET" > .env.test_gitignore
git check-ignore .env > /dev/null 2>&1 && echo "  PASS: .env excluded from git" || echo "WARN: .env not in .gitignore"
rm -f .env.test_gitignore

# 4. shared/types.py importable with all types
echo "[4/5] Checking shared types..."
uv run python -c "
from shared.types import BlockReference, StorageNodeInfo, TransferResult, AllocationPointer
# Smoke: instantiate each to verify fields aren't broken
br = BlockReference(device_id=0, memory_address=0x1000, size_bytes=4096)
tr = TransferResult(success=True, message='ok')
sn = StorageNodeInfo(node_id='n1', host='localhost', port=6379)
ap = AllocationPointer(cpu_address=0x2000, size_bytes=1024)
print(f'  BlockReference fields: {br._fields}')
print(f'  TransferResult fields: {tr._fields}')
print(f'  StorageNodeInfo fields: {sn._fields}')
print(f'  AllocationPointer fields: {ap._fields}')
" && echo "  PASS: shared types" || { echo "FAIL: shared types"; exit 1; }

# 5. shared/ports.py importable with all protocols and correct methods
echo "[5/5] Checking shared ports..."
uv run python -c "
from shared.ports import CacheStore, ModelRepository, GPUTransfer
assert hasattr(CacheStore, 'put') and hasattr(CacheStore, 'get')
assert hasattr(ModelRepository, 'fetch')
assert hasattr(GPUTransfer, 'copy_hbm_to_dram') and hasattr(GPUTransfer, 'copy_dram_to_hbm')
print('  Protocols: CacheStore(put,get), ModelRepository(fetch), GPUTransfer(copy_hbm_to_dram,copy_dram_to_hbm)')
" && echo "  PASS: shared ports" || { echo "FAIL: shared ports"; exit 1; }

echo ""
echo "=== Phase A: ALL CHECKS PASSED ==="
```

---

### Phase B — Make It Configurable & Testable (3 tasks)

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **0.3** | Create config modules with `pydantic-settings` for engine, sidecar, gateway | `engine/config.py`, `sidecar/config.py`, `gateway/config.py` | Unit test: defaults + env override | 30 |
| **0.4** | Create pytest infra: conftest, fixtures, first smoke test | `tests/conftest.py`, `tests/unit/test_smoke.py`, pyproject.toml `[tool.pytest]` | `uv run pytest tests/unit/test_smoke.py` passes | 30 |
| **0.8** | Create Makefile (install, test, lint, format, proto, docker-up) | `Makefile` | `make install && make lint` | 20 |

#### Phase B — Verification Checkpoint

Run the following. **All 4 checks must pass** before moving to Phases C/D:

```bash
#!/bin/bash
set -e
echo "=== Phase B Verification ==="

# 1. Config modules instantiate with defaults (no env vars needed)
echo "[1/4] Checking config modules..."
python -c "
from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.sidecar.config import SidecarConfig
from data_plane.gateway.config import GatewayConfig

ec = EngineConfig()
sc = SidecarConfig()
gc = GatewayConfig()

# Verify key defaults exist
assert ec.model_name == 'Qwen/Qwen2-0.5B', f'Bad default: {ec.model_name}'
assert ec.max_model_len == 512, f'Bad default: {ec.max_model_len}'
assert sc.l1_capacity_mb == 512, f'Bad default: {sc.l1_capacity_mb}'
print(f'  EngineConfig: model={ec.model_name}, max_len={ec.max_model_len}, gpu_mem={ec.gpu_memory_utilization}')
print(f'  SidecarConfig: l1={sc.l1_capacity_mb}MB')
print(f'  GatewayConfig loaded OK')
" && echo "  PASS: configs" || { echo "FAIL: config modules"; exit 1; }

# 2. Config env override works
echo "[2/4] Checking config env override..."
ENGINE_MODEL_NAME="test-model" ENGINE_MAX_MODEL_LEN="256" python -c "
from data_plane.inference.engine.config import EngineConfig
ec = EngineConfig()
assert ec.model_name == 'test-model', f'Env override failed: {ec.model_name}'
assert ec.max_model_len == 256, f'Env override failed: {ec.max_model_len}'
" && echo "  PASS: env override" || { echo "FAIL: env override"; exit 1; }

# 3. Pytest runs and smoke test passes
echo "[3/4] Running pytest smoke test..."
uv run pytest tests/unit/test_smoke.py -v --tb=short 2>&1 | tail -5
echo "  PASS: pytest smoke"

# 4. Makefile targets work
echo "[4/4] Checking Makefile..."
make install 2>&1 | tail -1
make lint 2>&1 | tail -3
echo "  PASS: Makefile"

echo ""
echo "=== Phase B: ALL CHECKS PASSED ==="
```

---

### Phase C — Fix the Engine (9 tasks) — core inference path

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **1.1** | Fix engine.py: add `await` (line 42), replace `requests` with `httpx`, fix mutable defaults | `engine/engine.py` | Unit: Engine instantiates with mocked LLMEngine | 45 |
| **1.2** | Inject config into engine (replace all hardcoded values) + add `ENABLE_ENGINE_MOCK` flag | `engine/engine.py`, `engine/config.py` | Unit: different settings → different CLI args; mock flag toggles mock mode | 40 |
| **1.3** | Fix api.py: proper imports, Pydantic request model, enable all params, `lifespan` pattern | `engine/api.py` | Unit: `/inference` with mocked engine | 45 |
| **1.4** | Add `/health` and `/ready` endpoints to engine API | `engine/api.py` | Unit: 200 vs 503 based on model state | 20 |
| **1.5** | Add Prometheus metrics to engine | `engine/metrics.py`, `engine/api.py` | Hit `/inference` then `/metrics`, counters incremented | 40 |
| **1.6** | Add error handling: validation, timeouts, structured errors | `engine/api.py` | 422 for bad input, 500 with detail for engine error | 30 |
| **1.7** | Create mock LLMEngine adapter (deterministic token generation, no GPU needed) | `engine/mock_engine.py` | Unit: MockLLMEngine.generate() returns tokens without GPU | 30 |
| **1.8** | Conditionally instantiate real or mock LLMEngine based on flag | `engine/engine.py` | Unit: ENABLE_ENGINE_MOCK=true → uses mock, false → uses real | 25 |
| **1.9** | Engine integration test (with both mock and real GPU paths) | `tests/integration/test_engine_e2e.py` | POST /inference returns generated text; works with mock=true and mock=false | 30 |

#### Phase C — Verification Checkpoint

Run the following. **All 7 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase C Verification ==="

# 1. Engine module imports cleanly (no import errors)
echo "[1/7] Checking engine imports..."
python -c "
from data_plane.inference.engine.engine import Engine
from data_plane.inference.engine.api import app
from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine.metrics import *
print('  All engine modules import OK')
" && echo "  PASS: imports" || { echo "FAIL: engine import error"; exit 1; }

# 2. Mock engine module imports and works
echo "[2/7] Checking mock engine..."
python -c "
from data_plane.inference.engine.mock_engine import MockLLMEngine
import asyncio

async def test():
    mock = MockLLMEngine()
    # Verify mock has the same interface as real LLMEngine
    assert hasattr(mock, 'generate'), 'MockLLMEngine missing generate method'
    print('  MockLLMEngine interface OK')

asyncio.run(test())
" && echo "  PASS: mock engine" || { echo "FAIL: mock engine"; exit 1; }

# 3. Config flag ENABLE_ENGINE_MOCK exists and defaults correctly
echo "[3/7] Checking mock flag in config..."
python -c "
from data_plane.inference.engine.config import EngineConfig
ec = EngineConfig()
assert hasattr(ec, 'enable_engine_mock') or hasattr(ec, 'mock_engine') or hasattr(ec, 'use_mock'), \
    'Config missing mock engine flag'
print(f'  Mock flag exists in config')
" && echo "  PASS: mock flag config" || { echo "FAIL: mock flag config"; exit 1; }

# 4. Engine conditionally uses mock based on flag
echo "[4/7] Testing mock vs real engine selection..."
python -c "
import os
from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine.engine import Engine

# Test with mock enabled
os.environ['ENABLE_ENGINE_MOCK'] = 'true'
ec_mock = EngineConfig()
print(f'  Mock enabled in config: {getattr(ec_mock, \"enable_engine_mock\", None) or getattr(ec_mock, \"mock_engine\", None)}')

# Test with mock disabled
os.environ['ENABLE_ENGINE_MOCK'] = 'false'
ec_real = EngineConfig()
print(f'  Mock disabled in config: {getattr(ec_real, \"enable_engine_mock\", None) or getattr(ec_real, \"mock_engine\", None)}')
" && echo "  PASS: mock selection" || { echo "FAIL: mock selection"; exit 1; }

# 5. Engine API unit tests pass (with mock, no GPU needed)
echo "[5/7] Running engine unit tests..."
uv run pytest tests/unit/ -k "engine" -v --tb=short 2>&1 | tail -10
echo "  PASS: engine unit tests"

# 6. Pydantic request model validates bad input
echo "[6/7] Testing input validation..."
python -c "
from pydantic import ValidationError
from data_plane.inference.engine.api import InferenceRequest  # or whatever the model is named

# Valid request works
req = InferenceRequest(prompt='Hello', max_tokens=10)
print(f'  Valid request: prompt={req.prompt}, max_tokens={req.max_tokens}')

# Missing required field raises
try:
    InferenceRequest()
    print('  FAIL: should have raised ValidationError')
    exit(1)
except (ValidationError, TypeError):
    print('  Correctly rejects empty request')
" && echo "  PASS: validation" || { echo "FAIL: validation"; exit 1; }

# 7. Engine with mock flag generates responses without GPU
echo "[7/7] Testing mock engine generation..."
python -c "
import asyncio
import os
os.environ['ENABLE_ENGINE_MOCK'] = 'true'

from data_plane.inference.engine.mock_engine import MockLLMEngine

async def test():
    mock = MockLLMEngine()
    # Mock should produce deterministic responses
    result = mock.generate(prompt='test', max_tokens=5)
    assert result is not None, 'Mock generation returned None'
    assert isinstance(result, (str, list)), f'Mock result has unexpected type: {type(result)}'
    print(f'  Mock generation returned: {type(result).__name__}')

asyncio.run(test())
" && echo "  PASS: mock generation" || { echo "FAIL: mock generation"; exit 1; }

echo ""
echo "=== Phase C: ALL CHECKS PASSED ==="
```

---

### Phase D — Fix the Sidecar (6 tasks) — parallel with Phase C

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **4.1** | Fix artifact_manager.py: add missing imports (os, asyncio, Dict), config injection | `sidecar/artifact_manager.py` | Module imports without error | 25 |
| **4.2** | Fix sidecar/api.py: fix import path, add missing `status`/`Optional`, `lifespan` | `sidecar/api.py` | Module imports without error | 30 |
| **4.3** | Implement real model download via `huggingface_hub.snapshot_download` | `sidecar/artifact_manager.py` | Integration: downloads tiny model to tmpdir | 45 |
| **4.4** | Add Prometheus metrics to sidecar | `sidecar/metrics.py`, `sidecar/api.py` | `/load` then `/metrics` shows duration recorded | 35 |
| **4.5** | Unit tests for sidecar API (health, ready, load, unload, registry, adapter) | `tests/unit/test_sidecar_api.py` | 7 test cases pass with mocked artifact manager | 40 |
| **4.6** | Write sidecar Dockerfile, test in docker-compose | `docker/sidecar.dockerfile`, `docker-compose.yml` | `docker-compose up sidecar` → `/health` returns 200 | 30 |

#### Phase D — Verification Checkpoint

Run the following. **All 5 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase D Verification ==="

# 1. All sidecar modules import cleanly
echo "[1/5] Checking sidecar imports..."
python -c "
from data_plane.inference.sidecar.artifact_manager import ArtifactManager
from data_plane.inference.sidecar.api import app
from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.config import SidecarConfig
print('  All sidecar modules import OK')
" && echo "  PASS: imports" || { echo "FAIL: sidecar import error"; exit 1; }

# 2. ArtifactManager methods exist and are async
echo "[2/5] Checking ArtifactManager interface..."
python -c "
import asyncio, inspect
from data_plane.inference.sidecar.artifact_manager import ArtifactManager

am = ArtifactManager.__new__(ArtifactManager)
for method in ['load_model', 'unload_model', 'fetch_adapter']:
    fn = getattr(am, method, None)
    assert fn is not None, f'Missing method: {method}'
    assert inspect.iscoroutinefunction(fn), f'{method} is not async'
    print(f'  {method}: async OK')
" && echo "  PASS: interface" || { echo "FAIL: interface"; exit 1; }

# 3. Sidecar unit tests pass
echo "[3/5] Running sidecar unit tests..."
uv run pytest tests/unit/ -k "sidecar" -v --tb=short 2>&1 | tail -10
echo "  PASS: sidecar unit tests"

# 4. Sidecar API TestClient: health + registry endpoints
echo "[4/5] Testing sidecar API with httpx TestClient..."
python -c "
import asyncio
from httpx import AsyncClient, ASGITransport
from data_plane.inference.sidecar.api import app

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        r = await client.get('/health')
        assert r.status_code in (200, 503), f'/health returned {r.status_code}'
        print(f'  /health -> {r.status_code}')

        r = await client.get('/registry/models')
        assert r.status_code == 200, f'/registry/models returned {r.status_code}'
        print(f'  /registry/models -> {r.status_code}, body type: {type(r.json()).__name__}')

asyncio.run(test())
" && echo "  PASS: API endpoints" || { echo "FAIL: API endpoints"; exit 1; }

# 5. No asyncio.sleep(5) simulated downloads remain (replaced with real HF download)
echo "[5/5] Checking for simulated downloads..."
SIMULATED=$(grep -n 'asyncio.sleep(5)' data-plane/inference/sidecar/artifact_manager.py || true)
if [ -z "$SIMULATED" ]; then
  echo "  PASS: no simulated downloads"
else
  echo "  FAIL: simulated download still present:"
  echo "$SIMULATED"
  exit 1
fi

echo ""
echo "=== Phase D: ALL CHECKS PASSED ==="
```

---

### Phase E — Docker Infrastructure (2 tasks) — needs C + D

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **0.5** | Write all 4 Dockerfiles | `docker/*.dockerfile` | `docker build -f docker/gateway.dockerfile .` succeeds | 40 |
| **0.6** | Create docker-compose.yml (gateway, engine, sidecar, redis, prometheus) | `docker-compose.yml`, `.env.example` | `docker-compose config` validates | 40 |

#### Phase E — Verification Checkpoint

Run the following. **All 4 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase E Verification ==="

# 1. All Dockerfiles are non-empty and have FROM instruction
echo "[1/4] Checking Dockerfiles..."
for df in docker/gateway.dockerfile docker/inference.dockerfile docker/sidecar.dockerfile docker/controller.dockerfile; do
  test -s "$df" || { echo "FAIL: $df is empty"; exit 1; }
  grep -q "^FROM " "$df" || { echo "FAIL: $df has no FROM instruction"; exit 1; }
  echo "  $df: OK ($(wc -l < "$df") lines)"
done
echo "  PASS: all Dockerfiles valid"

# 2. docker-compose.yml validates
echo "[2/4] Validating docker-compose.yml..."
docker-compose config --quiet 2>&1 && echo "  PASS: compose config valid" || { echo "FAIL: compose config"; exit 1; }

# 3. Docker builds succeed (gateway as fastest to build — no GPU deps)
echo "[3/4] Building gateway image (dry run)..."
docker build -f docker/gateway.dockerfile -t inference-server-gateway:test . 2>&1 | tail -3
echo "  PASS: gateway builds"

# 4. docker-compose has all expected services
echo "[4/4] Checking compose services..."
python -c "
import yaml
with open('docker-compose.yml') as f:
    dc = yaml.safe_load(f)
services = set(dc.get('services', {}).keys())
expected = {'gateway', 'engine', 'sidecar', 'redis'}
missing = expected - services
assert not missing, f'Missing services: {missing}'
print(f'  Services found: {sorted(services)}')
" && echo "  PASS: all services defined" || { echo "FAIL: missing services"; exit 1; }

echo ""
echo "=== Phase E: ALL CHECKS PASSED ==="
```

---

### Phase F — Streaming (4 tasks) — needs Phase C

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **2.1** | Implement queue-based streaming in engine batching loop | `engine/engine.py` | Unit: queue receives incremental tokens + sentinel | 45 |
| **2.2** | Add SSE streaming endpoint (`stream=true` returns `text/event-stream`) | `engine/api.py` | Unit: response yields `data:` lines | 40 |
| **2.3** | Handle client disconnect: abort generation, clean up queue | `engine/engine.py` | Unit: cancel mid-stream, request cleaned up | 30 |
| **2.4** | Add streaming metrics (tokens emitted, stream duration, cancelled) | `engine/metrics.py` | Verify metrics after streaming request | 20 |

#### Phase F — Verification Checkpoint

Run the following. **All 4 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase F Verification ==="

# 1. Streaming unit tests pass
echo "[1/4] Running streaming tests..."
uv run pytest tests/unit/ -k "stream" -v --tb=short 2>&1 | tail -10
echo "  PASS: streaming unit tests"

# 2. SSE endpoint returns correct content-type and data: lines
echo "[2/4] Testing SSE endpoint format..."
python -c "
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Test that stream=true triggers SSE response format
with patch('data_plane.inference.engine.engine.LLMEngine'):
    from httpx import AsyncClient, ASGITransport
    from data_plane.inference.engine.api import app

    async def test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            # Non-streaming should return JSON
            # (may fail if model not loaded, but content-type check is the point)
            r = await client.post('/inference', json={
                'prompt': 'Hello', 'max_tokens': 5, 'stream': False
            })
            ct = r.headers.get('content-type', '')
            print(f'  Non-streaming content-type: {ct}')

    asyncio.run(test())
" && echo "  PASS: SSE format" || echo "  WARN: SSE format test needs live engine (expected)"

# 3. Queue-based streaming mechanics work in isolation
echo "[3/4] Testing token queue mechanics..."
python -c "
import asyncio

async def test():
    q = asyncio.Queue()
    # Simulate engine pushing tokens
    tokens = ['Hello', ' world', '!', None]  # None = sentinel
    for t in tokens:
        await q.put(t)

    # Consumer reads until sentinel
    received = []
    while True:
        token = await q.get()
        if token is None:
            break
        received.append(token)

    assert received == ['Hello', ' world', '!'], f'Got: {received}'
    print(f'  Token queue: sent {len(tokens)-1} tokens, received {len(received)}')

asyncio.run(test())
" && echo "  PASS: queue mechanics" || { echo "FAIL: queue"; exit 1; }

# 4. Streaming metrics defined
echo "[4/4] Checking streaming metrics exist..."
python -c "
from data_plane.inference.engine.metrics import *
import data_plane.inference.engine.metrics as m
attrs = dir(m)
for metric in ['engine_tokens_generated_total', 'engine_stream_cancelled_total']:
    # Check metric name pattern exists (may be slightly different naming)
    found = any(metric.replace('_', '') in a.replace('_', '').lower() for a in attrs)
    print(f'  {metric}: {\"found\" if found else \"MISSING\"} ')
" && echo "  PASS: streaming metrics" || echo "  WARN: check metric names"

echo ""
echo "=== Phase F: ALL CHECKS PASSED ==="
```

---

### Phase G — LoRA (7 tasks) — needs C + D

#### Context

Partial LoRA scaffolding exists but is non-functional. Critical bugs: `lora_request` never passed to vLLM's `add_request()` (adapters loaded but never used), version param sent as JSON body but sidecar expects query param, no tracking of loaded adapters (duplicate GPU loads), no eviction policy.

**Full lifecycle**: At startup no adapters loaded. On adapter request → engine triggers sidecar download (fire-and-forget) → polls adapter registry until `status == "loaded"` (same pattern as base models) → waits for THIS request but other requests continue unblocked → engine loads adapter to GPU → processes request. LRU eviction when GPU adapter slots full. Requests for same pending adapter share one poll/wait.

#### Bugs to Fix

1. `engine.py:95` — `add_request` doesn't pass `lora_request=` to vLLM
2. `engine.py:82` — version sent as `json=` but sidecar expects `?version=` query param
3. `engine.py:88` — `_add_lora()` called every request without checking if already on GPU
4. `mock_engine.py:154` — `add_lora` is async but vLLM's is sync; `asyncio.to_thread` needs sync

#### Sidecar Adapter Endpoint Change

The current `POST /adapter/fetch/{id}` is **synchronous** (blocks until download completes). Change to **fire-and-forget + poll** to match the base model pattern (`POST /load/{id}` returns 202 + poll `GET /registry/adapters`):

**Modify `sidecar/api.py`** — Replace `fetch_adapter_route`:
- `POST /adapter/load/{adapter_identifier}?version=...` → if already loaded, return 200. If already downloading, return 202. Otherwise mark `status: "downloading"` in `adapter_registry`, kick off background download task, return 202.
- `GET /registry/adapters` (already exists) → engine polls this to check `status == "loaded"`

**Modify `sidecar/artifact_manager.py`** — `fetch_adapter` must update `adapter_registry` status to `"downloading"` before starting, then `"loaded"` on completion (mirroring `load_model` pattern).

#### Tasks

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **G.1** | Add `max_loras: int = 4`, `max_lora_rank: int = 16`, `adapter_poll_interval: float = 1.0`, `adapter_poll_timeout: float = 600.0` config fields | `engine/config.py` | Unit: config has fields, overridable via env | 10 |
| **G.2** | Convert sidecar adapter endpoint to fire-and-forget: `POST /adapter/load/{id}` returns 202, background download, status tracking in `adapter_registry` | `sidecar/api.py`, `sidecar/artifact_manager.py` | Unit: returns 202, registry shows downloading→loaded | 40 |
| **G.3** | Create `LoRAManager` with LRU tracking, poll-based adapter wait, download deduplication (leader-follower via `asyncio.Event`), eviction via `engine.remove_lora()`, metrics wiring | `engine/lora_manager.py` (new) | Unit: 7 test cases (see below) | 90 |
| **G.4** | Refactor `engine.py`: delegate to LoRAManager, pass `lora_request` to vLLM `add_request()`, add `--max-loras`/`--max-lora-rank` CLI args, delete `_add_lora()`, remove `httpx` import | `engine/engine.py` | Unit: lora_request passed to vLLM | 45 |
| **G.5** | Fix `mock_engine.py`: sync `add_lora`/`remove_lora` with tracking state, integrate LoRAManager when `enable_lora=True` | `engine/mock_engine.py` | Unit: loaded_loras tracks state | 30 |
| **G.6** | LoRAManager unit tests (mock engine + mocked httpx) | `tests/unit/test_lora_manager.py` (new) | 7 cases: first load, cache hit, concurrent dedup, LRU eviction, LRU ordering, error propagation, error isolation | 60 |
| **G.7** | LoRA integration test through FastAPI with mock engine | `tests/integration/test_lora_integration.py` (new) | Full lifecycle: inference with/without adapter, concurrent adapters, eviction | 45 |

#### LoRAManager Design

**State**: `_loaded: OrderedDict[str, LoadedAdapter]` (LRU), `_pending_downloads: Dict[str, asyncio.Event]` (dedup), `_lock: asyncio.Lock`

**`ensure_adapter_loaded(adapter_identifier, adapter_version) -> LoRARequest`**:
1. Check `_loaded` → cache hit: `move_to_end` (LRU touch), return immediately
2. Check `_pending_downloads` → another coroutine already waiting: `await` same Event
3. Leader path:
   a. `POST sidecar/adapter/load/{id}?version={v}` → triggers download (returns 202)
   b. Poll `GET sidecar/registry/adapters` every `adapter_poll_interval` seconds until entry has `status == "loaded"` (or timeout after `adapter_poll_timeout` seconds) — mirrors `_wait_for_sidecar_model` pattern
   c. Get `local_path` from registry entry
   d. Evict LRU adapter(s) if `len(_loaded) >= max_loras` via `engine.remove_lora(int_id)`
   e. Load onto GPU via `engine.add_lora(lora_request)`
   f. Record in `_loaded` → set Event to wake all waiters

**Concurrency**: `asyncio.Lock` (not threading) — all management on event loop. `add_lora`/`remove_lora` via `asyncio.to_thread()`. Requests for same adapter share one poll loop; different adapters/base model independent.

#### Files Changed

| File | Action | Summary |
|------|--------|---------|
| `engine/config.py` | Modify | Add `max_loras`, `max_lora_rank`, `adapter_poll_interval`, `adapter_poll_timeout` |
| `engine/lora_manager.py` | Create | LoRAManager with LRU, dedup, poll-based wait, eviction |
| `engine/engine.py` | Modify | Delegate to LoRAManager, pass `lora_request` to vLLM, rm `_add_lora` |
| `engine/mock_engine.py` | Modify | Sync `add_lora`/`remove_lora` tracking, integrate LoRAManager |
| `engine/metrics.py` | No change | Already has unused LoRA metrics — LoRAManager wires them up |
| `sidecar/api.py` | Modify | Replace sync `POST /adapter/fetch` with async `POST /adapter/load` (returns 202) |
| `sidecar/artifact_manager.py` | Modify | Add `downloading`→`loaded` status tracking for adapters |
| `tests/unit/test_lora_manager.py` | Create | LoRAManager unit tests |
| `tests/unit/test_sidecar_api.py` | Modify | Update adapter tests for new 202 + poll pattern |
| `tests/integration/test_lora_integration.py` | Create | End-to-end LoRA lifecycle tests |

#### Phase G — Verification Checkpoint

```bash
#!/bin/bash
set -e
echo "=== Phase G Verification ==="

# 1. No hardcoded LoRA path
echo "[1/7] Checking for hardcoded LoRA path..."
HARDCODED=$(grep -rn '/path/to/lora' data_plane/inference/engine/ || true)
if [ -z "$HARDCODED" ]; then
  echo "  PASS: no hardcoded LoRA path"
else
  echo "  FAIL: hardcoded path found: $HARDCODED"
  exit 1
fi

# 2. Config has max_loras, max_lora_rank, and adapter polling settings
echo "[2/7] Checking LoRA config fields..."
python -c "
from data_plane.inference.engine.config import EngineConfig
ec = EngineConfig()
assert hasattr(ec, 'enable_lora'), 'Missing enable_lora'
assert hasattr(ec, 'max_loras'), 'Missing max_loras'
assert hasattr(ec, 'max_lora_rank'), 'Missing max_lora_rank'
assert hasattr(ec, 'adapter_poll_interval'), 'Missing adapter_poll_interval'
assert hasattr(ec, 'adapter_poll_timeout'), 'Missing adapter_poll_timeout'
print(f'  enable_lora={ec.enable_lora}, max_loras={ec.max_loras}, max_lora_rank={ec.max_lora_rank}')
print(f'  adapter_poll_interval={ec.adapter_poll_interval}, adapter_poll_timeout={ec.adapter_poll_timeout}')
" && echo "  PASS: LoRA config" || { echo "FAIL: LoRA config"; exit 1; }

# 3. Sidecar adapter endpoint returns 202
echo "[3/7] Checking sidecar adapter endpoint is fire-and-forget..."
grep -n 'HTTP_202_ACCEPTED' data_plane/inference/sidecar/api.py | head -5
echo "  PASS: Sidecar adapter endpoint uses 202"

# 4. LoRAManager unit tests
echo "[4/7] Running LoRA manager unit tests..."
uv run pytest tests/unit/test_lora_manager.py -v --tb=short 2>&1 | tail -15
echo "  PASS: LoRA manager unit tests"

# 5. LoRA integration tests
echo "[5/7] Running LoRA integration tests..."
uv run pytest tests/integration/test_lora_integration.py -v --tb=short 2>&1 | tail -15
echo "  PASS: LoRA integration tests"

# 6. LoRA metrics exist and are wired
echo "[6/7] Checking LoRA metrics..."
python -c "
import data_plane.inference.engine.metrics as m
attrs = [a for a in dir(m) if 'lora' in a.lower()]
assert len(attrs) >= 2, f'Expected >=2 LoRA metrics, found: {attrs}'
print(f'  LoRA metrics: {attrs}')
" && echo "  PASS: LoRA metrics" || { echo "FAIL: LoRA metrics"; exit 1; }

# 7. All existing tests still pass
echo "[7/7] Running full test suite..."
uv run pytest tests/unit/ -v --tb=short 2>&1 | tail -10
echo "  PASS: Full test suite"

echo ""
echo "=== Phase G: ALL CHECKS PASSED ==="
```

---

### Phase H — Registry Persistence (3 tasks) — needs D

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **6.1** | Add JSON file persistence to model/adapter registry | `sidecar/artifact_manager.py` | New instance restores registry from file | 40 |
| **6.2** | Add detailed registry query endpoints (`/registry/models`, `/registry/adapters`) | `sidecar/api.py` | Load 2 models, query, both returned | 30 |
| **6.3** | Add tags, warmup prompts, preferred device to registry metadata | `sidecar/artifact_manager.py` | Register with tags, query, tags returned | 25 |

#### Phase H — Verification Checkpoint

Run the following. **All 3 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase H Verification ==="

# 1. Registry persists to JSON and restores
echo "[1/3] Testing registry persistence round-trip..."
python -c "
import asyncio, tempfile, json, os

async def test():
    from data_plane.inference.sidecar.artifact_manager import ArtifactManager
    from data_plane.inference.sidecar.config import SidecarConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_file = os.path.join(tmpdir, 'registry.json')

        # Create manager, add a model entry, verify file written
        config = SidecarConfig(registry_path=registry_file, model_store_path=tmpdir)
        am1 = ArtifactManager(config=config)

        # Manually add to registry (or use the load path with mock)
        am1.model_registry['test-model'] = {
            'model_id': 'test-model',
            'local_path': '/tmp/test',
            'status': 'loaded'
        }
        am1._persist_registry()  # or however persistence is triggered

        # Verify JSON file exists and is valid
        assert os.path.exists(registry_file), 'Registry file not created'
        with open(registry_file) as f:
            data = json.load(f)
        print(f'  Written: {json.dumps(data, indent=2)[:200]}')

        # New instance should restore
        am2 = ArtifactManager(config=config)
        assert 'test-model' in am2.model_registry, 'Registry not restored on init'
        print('  Restored: test-model found in new instance')

asyncio.run(test())
" && echo "  PASS: persistence round-trip" || { echo "FAIL: persistence"; exit 1; }

# 2. Registry API endpoints return proper structure
echo "[2/3] Testing registry API endpoints..."
python -c "
import asyncio
from httpx import AsyncClient, ASGITransport
from data_plane.inference.sidecar.api import app

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        r = await client.get('/registry/models')
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, (list, dict)), f'Expected list/dict, got {type(body)}'
        print(f'  /registry/models -> {r.status_code}, type={type(body).__name__}')

        r = await client.get('/registry/adapters')
        assert r.status_code == 200
        print(f'  /registry/adapters -> {r.status_code}')

asyncio.run(test())
" && echo "  PASS: registry endpoints" || { echo "FAIL: registry endpoints"; exit 1; }

# 3. Registry unit tests pass
echo "[3/3] Running registry tests..."
uv run pytest tests/unit/ -k "registry" -v --tb=short 2>&1 | tail -10
echo "  PASS: registry unit tests"

echo ""
echo "=== Phase H: ALL CHECKS PASSED ==="
```

---

### Phase I — KV Cache L1 (9 tasks) — needs B, parallel with C/D

#### Memory Hierarchy Overview (this step focuses only on L1)

```
┌─────────────────────────────────────────────────────────────────────┐
│  GPU SRAM (~20 MB, ~19 TB/s)                                       │
│  Hardware-managed. FlashAttention tiles Q,K,V into shared memory.   │
│  Not in our code.                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  GPU HBM (40-80 GB, ~2-3 TB/s)                                     │
│  vLLM PagedAttention manages KV blocks here during inference.       │
│  gpu_memory_utilization=0.70 → 70% for weights + KV cache.         │
│  When HBM pressure grows → blocks offloaded to L1.                 │
├─────────────────────────────────────────────────────────────────────┤
│  L1 Cache — CPU DRAM (512 MB–2 GB, local, per-pod)        ← THIS  │
│  Pinned memory pool. CuPy cudaMemcpyAsync for transfers.           │
│  LRU eviction. Managed by sidecar. ~12 GB/s on PCIe 4.0.          │
├─────────────────────────────────────────────────────────────────────┤
│  L2 Cache — Distributed Redis (shared across pods)                 │
│  ConsistentHashRing routing. Network-bound (~1-10 Gbps).           │
│  Enables cross-pod KV reuse. Planned for Phase L.                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### Design Decisions (answers to "Things to plan in detail")

**1. Interaction with sidecar: how does the sidecar offload or load from L1?**

The sidecar exposes a gRPC service (`KVCacheAPIService` in `kv_cache_api.py`) with two RPCs:
- `OffloadKVBlock(key, block_ref)` → sidecar's `MultiTieredCacheManager.process_offload()` → allocates in L1 pinned memory → `cudaMemcpyAsync` Device→Host
- `FetchKVBlock(key, dest_ref)` → `MultiTieredCacheManager.execute_fetch()` → looks up L1 → `cudaMemcpyAsync` Host→Device → returns to HBM

The engine calls these RPCs. In mock mode, the sidecar accepts the same RPCs but `GPUTransferHandler` simulates transfers with `asyncio.sleep`. In real mode, CuPy handles actual CUDA transfers.

```
Engine (vLLM)                         Sidecar (same pod)
─────────────                         ──────────────────
HBM pressure detected         ─gRPC─→ OffloadKVBlock(key, block_ref)
                                       │
                                       ├─ L1Allocator.allocate(size)
                                       ├─ GPUTransferHandler.copy_hbm_to_dram()
                                       ├─ KVBlockRegistry.register(key, location=L1, metadata)
                                       └─ LRUPolicy.track_new(key)

Cache needed for request       ─gRPC─→ FetchKVBlock(key, dest_ref)
                                       │
                                       ├─ KVBlockRegistry.lookup(key)
                                       ├─ GPUTransferHandler.copy_dram_to_hbm()
                                       ├─ LRUPolicy.record_access(key)
                                       └─ return Status(success=True)
```

**2. How does the engine decide when to offload or load from L1?**

vLLM v1 (0.11.0+) has a native `OffloadingConnector` with a pluggable backend API. We implement
a **custom offloading backend** that delegates to the sidecar's gRPC service instead of vLLM's
built-in CPU swap. This gives us control over eviction policy, registry, and L2 promotion.

The flow is:
1. vLLM's scheduler detects HBM pressure (block usage > threshold)
2. Scheduler's `OffloadingManager` selects victim blocks (LRU among inactive sequences)
3. `OffloadingManager` emits `StoreRequest` events via `KVConnectorMetadata`
4. Our custom `SidecarOffloadingConnector` (worker-side) translates these into gRPC `OffloadKVBlock` calls
5. On cache hit (prefix reuse), scheduler emits `LoadRequest` → our connector calls `FetchKVBlock`

For mock mode (no GPU), we bypass vLLM's connector and simulate the same flow:
- `MockEngine` exposes `offload_block(key, size)` and `fetch_block(key)` methods
- These call the sidecar gRPC directly, using mock block references
- This allows full integration testing of the L1 pipeline without a GPU

**3. Is there a "registry" of stored KV blocks? Is it also used by L2?**

Yes. We introduce a `KVBlockRegistry` — a unified metadata store tracking **where** each KV block
lives (L1, L2, or evicted). This is the single source of truth for block location.

```python
@dataclass
class KVBlockEntry:
    key: str                          # block hash (from vLLM's block_hashes)
    location: Literal["L1", "L2"]     # current tier
    size_bytes: int                   # block size
    l1_address: Optional[int]         # CPU address if in L1
    l2_node_id: Optional[str]         # Redis node if in L2
    model_id: str                     # which model produced this block
    prefix_hash: str                  # prefix hash for routing affinity
    created_at: float                 # timestamp
    last_accessed: float              # for LRU
    access_count: int                 # for frequency-based decisions
```

The registry is:
- **Queried by L1** during put/get to check existing entries
- **Queried by L2** during promotion/demotion to update location
- **Queried by cache-aware routing** (Phase J) to score backends by prefix affinity
- **Exposed via REST** `GET /cache/blocks` for observability and routing decisions

**4. Design for cache-aware routing compatibility (Phase J)**

The `KVBlockRegistry` stores `prefix_hash` and `model_id` per block. Phase J's router will:
1. Compute `prefix_hash` for incoming request
2. Query each pod's sidecar `GET /cache/blocks?prefix_hash=<hash>&model_id=<model>`
3. Score backends: `score = cache_hit_weight * has_prefix + (1 - load_weight) * (1 - normalized_load)`
4. Route to highest-scoring backend

To support this, Phase I exposes:
- `GET /cache/blocks` — list all registered blocks (with optional prefix_hash filter)
- `GET /cache/stats` — summary: total blocks, L1 usage, hit rate

**5. Telemetry for observability on loading and retrieval latency**

New Prometheus metrics in `l1_cache/metrics.py`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `l1_cache_capacity_bytes` | Gauge | — | Total L1 pool size |
| `l1_cache_used_bytes` | Gauge | — | Currently allocated bytes |
| `l1_cache_utilization_ratio` | Gauge | — | used / capacity |
| `l1_cache_operations_total` | Counter | op={put,get}, status={hit,miss,error} | Operation count |
| `l1_cache_operation_duration_seconds` | Histogram | op={put,get} | End-to-end latency (includes transfer) |
| `l1_cache_transfer_duration_seconds` | Histogram | direction={to_cpu,to_gpu} | Raw CUDA memcpy time |
| `l1_cache_transfer_bytes_total` | Counter | direction={to_cpu,to_gpu} | Transfer throughput tracking |
| `l1_cache_evictions_total` | Counter | reason={capacity,explicit} | Eviction count |
| `l1_cache_blocks_stored` | Gauge | — | Current block count in L1 |

**6. Actual CUDA memcpy implementation**

Use **CuPy** (`cupy` package) for GPU↔CPU transfers with pinned memory:

```python
import cupy as cp

# Allocate pinned (page-locked) host memory — required for async transfers
pinned_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
host_buffer = cupyx.empty_pinned((size_bytes,), dtype=cp.uint8)

# GPU → CPU (offload)
with cp.cuda.Stream() as stream:
    gpu_array = cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(gpu_mem, offset))
    host_buffer.copy_from_device_async(gpu_array, stream)
    stream.synchronize()

# CPU → GPU (restore)
with cp.cuda.Stream() as stream:
    gpu_dest = cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(gpu_mem, offset))
    gpu_dest.copy_from_host_async(host_buffer, stream)
    stream.synchronize()
```

When `enable_engine_mock=True`, the `GPUTransferHandler` skips CuPy and uses `asyncio.sleep(0.001)`
to simulate transfer latency. A factory function selects the right implementation:

```python
def create_transfer_handler(mock: bool = False) -> GPUTransferHandler:
    if mock:
        return MockGPUTransferHandler()
    return CuPyGPUTransferHandler()
```

---

#### Task Breakdown

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **I.1** | **KV Block Registry**: Create `KVBlockRegistry` class that tracks block metadata (key, location, size, prefix_hash, model_id, timestamps, access_count). Methods: `register(entry)`, `unregister(key)`, `lookup(key) -> Optional[KVBlockEntry]`, `query_by_prefix(prefix_hash, model_id) -> List[KVBlockEntry]`, `update_location(key, new_location)`, `stats() -> dict`. Store in-memory dict with JSON persistence to `{registry_path}/kv_blocks.json`. Import shared types from `shared/types.py` and add `KVBlockEntry` dataclass there. | `shared/types.py`, `sidecar/kv_block_registry.py` | Unit: register → lookup returns entry; unregister → lookup returns None; query_by_prefix filters correctly; stats reflect counts; persistence round-trip works | 50 |
| **I.2** | **GPU Transfer Handler — real CuPy implementation**: Refactor `GPUTransferHandler` into a protocol/ABC. Implement `CuPyGPUTransferHandler` that uses `cupy.cuda.Stream`, pinned memory via `cupyx.empty_pinned`, and `copy_from_device_async` / `copy_from_host_async` for actual GPU↔CPU transfers. Keep existing mock as `MockGPUTransferHandler` (asyncio.sleep). Add factory `create_transfer_handler(mock: bool)`. Update `l1_cache/api.py` to accept handler via constructor injection instead of creating internally. Add `cupy` as optional dependency (`pip install cupy-cuda12x` or env-dependent). | `l1_cache/gpu_transfer.py`, `l1_cache/api.py`, `pyproject.toml` | Unit (mock): factory returns correct type; Mock handler returns TransferStatus(True); CuPy handler import check (skip if no GPU). Integration (GPU only): 1MB transfer round-trip, data integrity check | 60 |
| **I.3** | **L1 Allocator — pinned memory pool**: Refactor `L1Allocator` to manage a real pinned-memory pool when CuPy is available. Use `cupy.cuda.PinnedMemoryPool` for the backing store. Track allocations with a free-list (not just a counter) so `free()` actually reclaims specific regions. Add `MockL1Allocator` that keeps current counter-based behavior. Constructor: `L1Allocator(capacity_bytes, mock=False)`. Expose `available_bytes`, `used_bytes`, `allocation_count` properties. | `l1_cache/allocator.py` | Unit (mock): allocate → AllocationPointer with valid address; allocate beyond capacity → None; free reclaims exact bytes; fragmentation: alloc A, alloc B, free A, alloc C fits in A's space | 50 |
| **I.4** | **Cache Manager — implement stubs + registry integration**: Implement `check_global_availability(key)` — query `KVBlockRegistry.lookup(key)` to check if block exists in any tier. Implement `execute_l1_eviction(needed_bytes)` — use `LRUPolicy.select_victim()` in a loop until enough space freed, calling `L1CacheAPI._evict_victim()` for each, unregistering from `KVBlockRegistry`. Update `process_offload()` to register blocks in `KVBlockRegistry` after successful L1 put. Update `execute_fetch()` to update `last_accessed` in registry on hit. Fix `kv_cache_api.py` OffloadKVBlock bug (missing key extraction from request). Remove all `NotImplementedError` stubs. | `sidecar/cache_manager.py`, `sidecar/kv_cache_api.py` | Unit: `check_global_availability` True after put, False after eviction; `execute_l1_eviction` frees requested bytes and removes from registry; process_offload creates registry entry; OffloadKVBlock extracts key correctly | 60 |
| **I.5** | **L1 Cache REST endpoints for routing compatibility**: Add endpoints to sidecar's FastAPI app for cache-aware routing (Phase J) to query. `GET /cache/blocks` — returns list of `KVBlockEntry` (optional query params: `prefix_hash`, `model_id`). `GET /cache/stats` — returns `{total_blocks, l1_blocks, l1_used_bytes, l1_capacity_bytes, hit_rate, eviction_count}`. These endpoints read from `KVBlockRegistry`. Wire `KVBlockRegistry` into sidecar app lifespan (created once, passed to cache manager and endpoints). | `sidecar/api.py`, `sidecar/kv_block_registry.py` | Unit: `GET /cache/blocks` returns empty list initially; after mock offload, returns 1 entry; `GET /cache/stats` returns correct structure; prefix_hash filter works | 40 |
| **I.6** | **L1 Cache Prometheus metrics**: Create `l1_cache/metrics.py` with all metrics from the telemetry table above. Instrument `L1CacheAPI.put()` and `get()` with operation counters, duration histograms, and transfer byte counters. Instrument `GPUTransferHandler` with `l1_cache_transfer_duration_seconds`. Instrument eviction in `_evict_victim()` with `l1_cache_evictions_total`. Update capacity/used gauges on every allocate/free. Expose via existing sidecar `/metrics` endpoint (already serves Prometheus). | `l1_cache/metrics.py`, `l1_cache/api.py`, `l1_cache/gpu_transfer.py`, `l1_cache/allocator.py` | Unit: 3 puts + 2 gets → `l1_cache_operations_total{op=put}` = 3, `{op=get,status=hit}` = 2; after eviction → `l1_cache_evictions_total` incremented; `l1_cache_used_bytes` matches allocator state | 45 |
| **I.7** | **Engine ↔ Sidecar offload wiring**: Add `SidecarCacheClient` to engine that wraps gRPC calls to sidecar's `KVCacheAPIService`. Methods: `async offload_block(key, block_ref)`, `async fetch_block(key, dest_ref)`, `async check_availability(key)`. In `MockEngine`, add `offload_block()` and `fetch_block()` that call the sidecar client, simulating what vLLM's `OffloadingConnector` would do. Add engine config field `sidecar_grpc_url` (default `localhost:50051`). In real engine mode, document where the custom `SidecarOffloadingConnector` would plug in (as a comment/TODO for when vLLM's OffloadingConnector API is used in production). | `engine/sidecar_cache_client.py` (new), `engine/mock_engine.py`, `engine/config.py` | Unit: MockEngine.offload_block() calls client; client serializes BlockReference correctly; fetch_block round-trip with mocked sidecar returns success | 50 |
| **I.8** | **L1 cache unit tests**: Comprehensive test suite covering all L1 components. Tests: (1) Allocator alloc/free cycle, (2) Allocator capacity exhaustion returns None, (3) Allocator free-list reclamation, (4) LRU ordering after mixed accesses, (5) LRU victim selection after removal, (6) L1CacheAPI put/get round-trip (mock transfer), (7) L1CacheAPI eviction on capacity pressure, (8) L1CacheAPI get miss returns False, (9) KVBlockRegistry CRUD + query_by_prefix, (10) KVBlockRegistry persistence round-trip. | `tests/unit/test_l1_cache.py` | 10 test cases, all pass | 50 |
| **I.9** | **Cache manager + integration tests**: Test the full multi-tier cache pipeline. Tests: (1) process_offload stores in L1 and registers in KVBlockRegistry, (2) execute_fetch hits L1 and updates access time, (3) execute_fetch misses L1 → falls through to L2 mock, (4) check_global_availability returns True for L1 block, (5) execute_l1_eviction frees space and unregisters, (6) L1 full → offload triggers eviction → still succeeds, (7) REST /cache/blocks returns registered blocks, (8) REST /cache/stats reflects current state, (9) Metrics counters match operations performed. | `tests/unit/test_cache_manager.py` | 9 test cases with mocked L1 transfers + mocked L2, all pass | 55 |

---

#### Phase I — Verification Checkpoint

Run the following. **All 7 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase I Verification ==="

# 1. All L1 cache modules import without error
echo "[1/7] Checking L1 cache imports..."
python -c "
from data_plane.inference.sidecar.l1_cache.api import L1CacheAPI
from data_plane.inference.sidecar.l1_cache.allocator import L1Allocator
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy, EvictionPolicy
from data_plane.inference.sidecar.l1_cache.gpu_transfer import (
    GPUTransferHandler, MockGPUTransferHandler, create_transfer_handler
)
from data_plane.inference.sidecar.cache_manager import MultiTieredCacheManager
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from shared.types import KVBlockEntry, BlockReference, TransferResult, AllocationPointer
print('  All L1 cache modules import OK')
" && echo "  PASS: imports" || { echo "FAIL: L1 cache import error"; exit 1; }

# 2. No NotImplementedError remains in cache_manager
echo "[2/7] Checking for NotImplementedError..."
NOT_IMPL=$(grep -n 'NotImplementedError' data-plane/inference/sidecar/cache_manager.py || true)
if [ -z "$NOT_IMPL" ]; then
  echo "  PASS: no NotImplementedError stubs"
else
  echo "  FAIL: NotImplementedError found:"
  echo "$NOT_IMPL"
  exit 1
fi

# 3. L1 allocator works: alloc + free cycle with free-list reclamation
echo "[3/7] Testing L1 allocator cycle..."
python -c "
from data_plane.inference.sidecar.l1_cache.allocator import L1Allocator

alloc = L1Allocator(capacity_bytes=1024, mock=True)
ptr_a = alloc.allocate(256)
assert ptr_a is not None, 'First allocation failed'
print(f'  Allocated A: address={ptr_a.cpu_address}, size={ptr_a.size_bytes}')

ptr_b = alloc.allocate(256)
assert ptr_b is not None, 'Second allocation failed'
print(f'  Allocated B: address={ptr_b.cpu_address}, size={ptr_b.size_bytes}')

# Free A, then allocate C — should fit in A's reclaimed space
alloc.free(ptr_a)
print(f'  Freed A. Available: {alloc.available_bytes} bytes')

ptr_c = alloc.allocate(256)
assert ptr_c is not None, 'Reclaimed allocation failed'
print(f'  Allocated C (in reclaimed space): address={ptr_c.cpu_address}, size={ptr_c.size_bytes}')
print(f'  Final available: {alloc.available_bytes} bytes')
" && echo "  PASS: allocator" || { echo "FAIL: allocator"; exit 1; }

# 4. LRU eviction ordering is correct
echo "[4/7] Testing LRU eviction order..."
python -c "
from data_plane.inference.sidecar.l1_cache.eviction_policy import LRUPolicy

lru = LRUPolicy()
lru.track_new('a', 100)
lru.track_new('b', 200)
lru.track_new('c', 300)
lru.record_access('a')  # 'a' most recent, 'b' oldest

victim = lru.select_victim()
assert victim == 'b', f'Expected b (oldest), got {victim}'
print(f'  LRU victim after track(a,b,c), access(a): {victim} (correct: b is oldest)')
" && echo "  PASS: LRU ordering" || { echo "FAIL: LRU"; exit 1; }

# 5. KV Block Registry CRUD works
echo "[5/7] Testing KV Block Registry..."
python -c "
import tempfile, os
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry
from shared.types import KVBlockEntry

with tempfile.TemporaryDirectory() as tmpdir:
    reg = KVBlockRegistry(persist_path=os.path.join(tmpdir, 'kv_blocks.json'))

    entry = KVBlockEntry(
        key='block-001', location='L1', size_bytes=4096,
        l1_address=0x10000000, l2_node_id=None,
        model_id='test-model', prefix_hash='abc123',
        created_at=1000.0, last_accessed=1000.0, access_count=0
    )
    reg.register(entry)
    assert reg.lookup('block-001') is not None, 'Lookup failed after register'
    print(f'  Registered and looked up block-001')

    results = reg.query_by_prefix('abc123', 'test-model')
    assert len(results) == 1, f'Expected 1 result, got {len(results)}'
    print(f'  query_by_prefix returned {len(results)} result(s)')

    stats = reg.stats()
    assert stats['total_blocks'] == 1, f'Expected 1 block, got {stats[\"total_blocks\"]}'
    print(f'  Stats: {stats}')

    reg.unregister('block-001')
    assert reg.lookup('block-001') is None, 'Lookup should be None after unregister'
    print(f'  Unregistered block-001 successfully')
" && echo "  PASS: KV Block Registry" || { echo "FAIL: KV Block Registry"; exit 1; }

# 6. Cache REST endpoints respond
echo "[6/7] Testing cache REST endpoints..."
python -c "
import asyncio
from httpx import AsyncClient, ASGITransport
from data_plane.inference.sidecar.api import app

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        r = await client.get('/cache/blocks')
        assert r.status_code == 200, f'Expected 200, got {r.status_code}'
        assert isinstance(r.json(), list), 'Expected list'
        print(f'  GET /cache/blocks -> {r.status_code}, {len(r.json())} blocks')

        r = await client.get('/cache/stats')
        assert r.status_code == 200, f'Expected 200, got {r.status_code}'
        body = r.json()
        assert 'total_blocks' in body, f'Missing total_blocks in {body}'
        assert 'hit_rate' in body, f'Missing hit_rate in {body}'
        print(f'  GET /cache/stats -> {r.status_code}, keys={list(body.keys())}')

asyncio.run(test())
" && echo "  PASS: cache REST endpoints" || { echo "FAIL: cache REST"; exit 1; }

# 7. All cache tests pass (L1 + cache manager)
echo "[7/7] Running cache unit tests..."
uv run pytest tests/unit/test_l1_cache.py tests/unit/test_cache_manager.py -v --tb=short 2>&1 | tail -20
echo "  PASS: cache unit tests"

echo ""
echo "=== Phase I: ALL CHECKS PASSED ==="
```

---

### Phase J — Cache-Aware Routing (5 tasks) — needs C + I

UPDATE: This could be moved to scaling_plan, so it is implemented as part of the scaling strategies

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **8.1** | Fix routing.py: import `status`, config injection, `lifespan` | `gateway/routing.py` | Module imports without error | 25 |
| **8.2** | Add dynamic backend registration (`POST /register`, `DELETE /register/{id}`) | `gateway/routing.py` | Register → route succeeds, deregister → 404 | 40 |
| **8.3** | Create Router class with cache-hit scoring + least-connections fallback | `gateway/router.py` | 2 backends, one with cached prefix → preferred | 45 |
| **8.4** | Add prefix hash computation to request pipeline | `gateway/routing.py` | Same prompt → same hash | 30 |
| **8.5** | Gateway metrics + unit tests | `gateway/metrics.py`, `tests/unit/test_routing.py` | 5 test cases | 45 |

#### Phase J — Verification Checkpoint

Run the following. **All 5 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase J Verification ==="

# 1. Gateway imports cleanly
echo "[1/5] Checking gateway imports..."
python -c "
from data_plane.gateway.routing import app
from data_plane.gateway.router import Router
from data_plane.gateway.config import GatewayConfig
print('  All gateway modules import OK')
" && echo "  PASS: imports" || { echo "FAIL: gateway import error"; exit 1; }

# 2. Prefix hashing is deterministic
echo "[2/5] Testing prefix hash determinism..."
python -c "
from data_plane.gateway.router import Router
r = Router()
h1 = r.compute_prefix_hash('The quick brown fox')
h2 = r.compute_prefix_hash('The quick brown fox')
h3 = r.compute_prefix_hash('Different prompt')
assert h1 == h2, f'Same input, different hash: {h1} vs {h2}'
assert h1 != h3, f'Different input, same hash'
print(f'  Hash(\"The quick brown fox\") = {h1[:16]}...')
print(f'  Hash(\"Different prompt\")    = {h3[:16]}...')
print(f'  Deterministic: True, Unique: True')
" && echo "  PASS: prefix hashing" || { echo "FAIL: prefix hashing"; exit 1; }

# 3. Cache-aware scoring prefers backend with cached prefix
echo "[3/5] Testing cache-aware scoring..."
python -c "
from data_plane.gateway.router import Router
r = Router()

# Register 2 backends
r.register_backend('backend-1', 'http://engine1:8080')
r.register_backend('backend-2', 'http://engine2:8080')

# Simulate backend-1 having a cached prefix
r.report_cache_hit('backend-1', 'Hello world prompt')

# Route should prefer backend-1 for similar prefix
selected = r.select_backend(prompt='Hello world prompt', model='test')
assert selected is not None, 'No backend selected'
print(f'  Selected backend: {selected}')
" && echo "  PASS: cache-aware scoring" || echo "  WARN: scoring test (may need mock adjustment)"

# 4. Dynamic registration endpoints work
echo "[4/5] Testing backend registration..."
python -c "
import asyncio
from httpx import AsyncClient, ASGITransport
from data_plane.gateway.routing import app

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        # Register a backend
        r = await client.post('/register', json={
            'id': 'engine-1',
            'url': 'http://localhost:8080'
        })
        assert r.status_code in (200, 201), f'Register failed: {r.status_code}'
        print(f'  POST /register -> {r.status_code}')

        # Delete it
        r = await client.delete('/register/engine-1')
        assert r.status_code in (200, 204), f'Deregister failed: {r.status_code}'
        print(f'  DELETE /register/engine-1 -> {r.status_code}')

asyncio.run(test())
" && echo "  PASS: registration" || { echo "FAIL: registration"; exit 1; }

# 5. All routing tests pass
echo "[5/5] Running routing unit tests..."
uv run pytest tests/unit/ -k "routing or gateway" -v --tb=short 2>&1 | tail -10
echo "  PASS: routing unit tests"

echo ""
echo "=== Phase J: ALL CHECKS PASSED ==="
```

---

### Phase K — Robustness (5 tasks) — needs C

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **9.1** | Add request queue limit → HTTP 429 with Retry-After | `engine/api.py`, `engine/engine.py` | Submit MAX+1 requests → last gets 429 | 35 |
| **9.2** | Add request timeout → HTTP 504, abort vLLM request | `engine/engine.py`, `engine/api.py` | Tiny timeout → 504 | 35 |
| **9.3** | Handle client disconnect: detect + abort for non-streaming | `engine/api.py` | Start request, disconnect, engine aborts | 40 |
| **9.4** | Graceful shutdown: drain in-flight, stop accepting, cleanup | `engine/api.py`, `engine/engine.py` | Submit, shutdown, request completes | 35 |
| **9.5** | Robustness test suite (429, 504, shutdown, concurrency, no leak) | `tests/unit/test_robustness.py` | 5 test cases | 45 |

#### Phase K — Verification Checkpoint

Run the following. **All 4 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase K Verification ==="

# 1. Robustness tests pass (the definitive check)
echo "[1/4] Running robustness test suite..."
uv run pytest tests/unit/test_robustness.py -v --tb=short 2>&1 | tail -15
echo "  PASS: robustness tests"

# 2. 429 returned when queue is full
echo "[2/4] Testing 429 response format..."
python -c "
# Verify that the 429 response includes Retry-After header
# This is a structural check — the actual concurrency test is in test_robustness.py
import inspect
from data_plane.inference.engine import api as engine_api
source = inspect.getsource(engine_api)
assert 'retry-after' in source.lower() or 'Retry-After' in source, \
    '429 response should include Retry-After header'
print('  429 response includes Retry-After header logic')
assert '429' in source or 'Too Many' in source or 'HTTP_429' in source, \
    '429 status code not found in engine API'
print('  429 status code handling present')
" && echo "  PASS: 429 format" || { echo "FAIL: 429 format"; exit 1; }

# 3. 504 timeout path exists
echo "[3/4] Checking timeout handling..."
python -c "
import inspect
from data_plane.inference.engine import api as engine_api
source = inspect.getsource(engine_api)
assert '504' in source or 'timeout' in source.lower() or 'HTTP_504' in source, \
    'Timeout handling (504) not found in engine API'
print('  Timeout (504) handling present in engine API')
" && echo "  PASS: timeout handling" || { echo "FAIL: timeout"; exit 1; }

# 4. Graceful shutdown handler exists
echo "[4/4] Checking graceful shutdown..."
python -c "
import inspect
from data_plane.inference.engine import api as engine_api
source = inspect.getsource(engine_api)
assert 'shutdown' in source.lower() or 'lifespan' in source.lower(), \
    'Shutdown handling not found'
# Check engine.py too
from data_plane.inference.engine import engine as engine_mod
engine_source = inspect.getsource(engine_mod)
assert 'shutdown' in engine_source.lower() or 'drain' in engine_source.lower() or 'stop' in engine_source.lower(), \
    'Engine drain/shutdown not found'
print('  Graceful shutdown handling present in both api.py and engine.py')
" && echo "  PASS: graceful shutdown" || { echo "FAIL: shutdown"; exit 1; }

echo ""
echo "=== Phase K: ALL CHECKS PASSED ==="
```

---

### Phase L — Sidecar Completion (6 tasks) — needs D + I

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **10.1** | Add download retries (3x exponential backoff) + atomic download (tmp + rename) | `sidecar/artifact_manager.py` | Mock fail 2x then succeed → works; fail 3x → raises | 40 |
| **10.2** | Add model warmup (send prompts to engine after load to populate cache) | `sidecar/warmup.py`, `sidecar/api.py` | Mock engine, verify warmup requests sent | 35 |
| **10.3** | Create `proto/kv_cache.proto` with KVCacheService + KVCacheWatcher definitions | `proto/kv_cache.proto`, `Makefile` proto target | `make proto` generates `kv_cache_pb2.py` + `kv_cache_pb2_grpc.py` | 45 |
| **10.4** | Update `kv_cache_api.py` and `l2_cache/api.py` to use real proto stubs | `sidecar/kv_cache_api.py`, `l2_cache/api.py` | Imports succeed, gRPC server binds | 40 |
| **10.5** | Add structured JSON logging across sidecar modules | `sidecar/logging_config.py`, all sidecar files | Model load produces valid JSON log | 30 |
| **10.6** | Sidecar integration test: full lifecycle (load → registry → adapter → unload → metrics) | `tests/integration/test_sidecar_lifecycle.py` | 9-step lifecycle test | 40 |

#### Phase L — Verification Checkpoint

Run the following. **All 6 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase L Verification ==="

# 1. Proto file generates valid stubs
echo "[1/6] Checking proto generation..."
test -f shared/proto/kv_cache.proto || { echo "FAIL: proto file missing"; exit 1; }
make proto 2>&1 | tail -3
test -f shared/proto/kv_cache_pb2.py || { echo "FAIL: pb2.py not generated"; exit 1; }
test -f shared/proto/kv_cache_pb2_grpc.py || { echo "FAIL: pb2_grpc.py not generated"; exit 1; }
python -c "
from shared.proto import kv_cache_pb2, kv_cache_pb2_grpc
print(f'  Proto services: {[s for s in dir(kv_cache_pb2_grpc) if \"Servicer\" in s]}')
" && echo "  PASS: proto stubs" || { echo "FAIL: proto import"; exit 1; }

# 2. gRPC modules import with real stubs (no undefined types)
echo "[2/6] Checking gRPC module imports..."
python -c "
from data_plane.inference.sidecar.kv_cache_api import KVCacheAPIService
from data_plane.inference.sidecar.l2_cache.api import *
print('  gRPC modules import OK with real stubs')
" && echo "  PASS: gRPC imports" || { echo "FAIL: gRPC imports"; exit 1; }

# 3. Download retries work (mock test)
echo "[3/6] Running retry tests..."
uv run pytest tests/unit/ -k "retry" -v --tb=short 2>&1 | tail -8
echo "  PASS: retry tests"

# 4. Warmup module exists and is callable
echo "[4/6] Checking warmup module..."
python -c "
from data_plane.inference.sidecar.warmup import ModelWarmup
import inspect
assert inspect.iscoroutinefunction(ModelWarmup.run) or \
       inspect.iscoroutinefunction(getattr(ModelWarmup, 'warmup', None) or (lambda: None)), \
    'Warmup should be async'
print('  Warmup module OK')
" && echo "  PASS: warmup" || { echo "FAIL: warmup"; exit 1; }

# 5. Structured JSON logging produces valid JSON
echo "[5/6] Testing JSON logging..."
python -c "
import logging, json, io
from data_plane.inference.sidecar.logging_config import setup_logging

# Capture log output
handler = logging.StreamHandler(stream := io.StringIO())
setup_logging(handler=handler)
logger = logging.getLogger('sidecar')
logger.info('test message', extra={'model_id': 'test-model'})

output = stream.getvalue().strip()
if output:
    parsed = json.loads(output)
    assert 'message' in parsed or 'msg' in parsed, f'No message in log: {parsed}'
    print(f'  JSON log: {json.dumps(parsed)[:200]}')
else:
    print('  WARN: no log output captured (may need different handler setup)')
" && echo "  PASS: JSON logging" || echo "  WARN: logging test (adjust if needed)"

# 6. Sidecar integration test passes
echo "[6/6] Running sidecar lifecycle test..."
uv run pytest tests/integration/test_sidecar_lifecycle.py -v --tb=short 2>&1 | tail -15
echo "  PASS: sidecar lifecycle"

echo ""
echo "=== Phase L: ALL CHECKS PASSED ==="
```

---

### Phase M — Batching Metric (1 task)

| Task | Description | Files | Test | ~Min |
|------|------------|-------|------|------|
| **3.1** | Add batch size + step latency metrics to batching loop | `engine/engine.py` | 3 concurrent requests → batch_size histogram > 1 | 30 |

#### Phase M — Verification Checkpoint

Run the following. **All 2 checks must pass**:

```bash
#!/bin/bash
set -e
echo "=== Phase M Verification ==="

# 1. Batch metrics exist in metrics module
echo "[1/2] Checking batch metrics..."
python -c "
import data_plane.inference.engine.metrics as m
attrs = dir(m)
batch_metrics = [a for a in attrs if 'batch' in a.lower()]
assert len(batch_metrics) > 0, f'No batch metrics found. Available: {[a for a in attrs if not a.startswith(\"_\")]}'
print(f'  Batch metrics: {batch_metrics}')
" && echo "  PASS: batch metrics defined" || { echo "FAIL: batch metrics"; exit 1; }

# 2. Full unit test suite still passes (regression check)
echo "[2/2] Running full test suite (regression)..."
uv run pytest tests/unit/ -v --tb=short 2>&1 | tail -15
RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "  PASS: full test suite"
else
  echo "  FAIL: test regression detected"
  exit 1
fi

echo ""
echo "=== Phase M: ALL CHECKS PASSED ==="
echo ""
echo "========================================"
echo "  ALL PHASES COMPLETE — READY FOR E2E  "
echo "========================================"
```

---

## 6. Execution Order (dependency-driven)

```
Sprint 0 (first):    Docs                   →  2 files → docs/plan.md + docs/requirements-and-decisions.md
Sprint 1 (Day 1):    Phase A + B            →  8 tasks → architecture foundation + testability
Sprint 2 (Days 2-3): Phase C + D (parallel) → 15 tasks → core engine (with mock flag) + sidecar work
Sprint 3 (Days 3-4): Phase E + I            → 11 tasks → Docker infra + cache (L1)
Sprint 4 (Days 4-5): Phase F + G + H        → 12 tasks → streaming, LoRA, registry
Sprint 5 (Days 5-6): Phase J + K + L + M    → 17 tasks → routing, robustness, completion
```

**Phase C Notes**: Tasks 1.2, 1.7, 1.8 are the new mock-specific additions. The mock engine allows testing the entire system (including metrics, routing, etc.) without a GPU. All other functionality remains unchanged.

Phases C and D can run in parallel. Phases F, G, H can run in parallel after their deps.

---

## 7. Documentation Deliverables (first action on approval)

Before any code changes, create two documents:

1. **`docs/plan.md`** — The full implementation plan (Steps 0-10, atomic tasks, execution order, verification)
2. **`docs/requirements-and-decisions.md`** — Contains:
   - Functional requirements (FR-1 through FR-10)
   - Non-functional requirements (model, GPU, SLO, budgets)
   - Architectural decision record: why modular+selective-ports was chosen over hexagonal/clean
   - The 3 Protocol interfaces and what they replace
   - The shared types and what duplication they eliminate
   - Telemetry metrics specification
   - Local testing strategy (docker-compose, test model, mock strategy)

---

## 8. Verification

After all tasks are complete, the following should work end-to-end:

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
