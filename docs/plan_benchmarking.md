# Benchmarking Plan

> Validate performance characteristics, identify bottlenecks, and establish regression baselines for the inference server.

## Table of Contents

1. [Overview](#1-overview)
2. [Tooling](#2-tooling)
3. [Benchmark Categories](#3-benchmark-categories)
4. [Metrics Collection Strategy](#4-metrics-collection-strategy)
5. [SLO Definitions and Compliance](#5-slo-definitions-and-compliance)
6. [Report Generation](#6-report-generation)
7. [Implementation Phases](#7-implementation-phases)
8. [Known Prerequisites](#8-known-prerequisites)

---

## 1. Overview

The inference server comprises three services (Gateway :8000, Engine :8080, Sidecar :8001) backed by vLLM, a Redis L2 cache, and Prometheus. All features through Phase M are implemented. This plan defines **9 benchmark categories** with concrete test scenarios, parameter matrices, and tooling to produce repeatable, comparable results.

**Goals:**
- Establish performance baselines for every feature in `docs/features.md`
- Detect regressions automatically via SLO compliance checks
- Identify bottlenecks under realistic production workloads
- Validate queue pressure behavior at `ENGINE_MAX_PENDING=10`

---

## 2. Tooling

| Tool | Role | Notes |
|------|------|-------|
| **Locust** | Load generation | Scriptable Python users, real-time web UI, CSV/JSON export |
| **httpx (async)** | Precision latency | Direct async client for single-request profiling and streaming measurement |
| **Prometheus API** | Server-side metrics | Query `http://localhost:9090/api/v1/query` for all engine/sidecar counters and histograms |
| **pynvml** | GPU monitoring | Real-time GPU utilization, memory, temperature during benchmark runs |
| **pytest-benchmark** | Micro-benchmarks | Optional — for unit-level latency of internal components |
| **Docker Compose** | Environment control | Deterministic multi-service startup with pinned resource limits |

### Standard Test Environment

```
GPU:            1x NVIDIA (≥16 GB VRAM)
Model:          Qwen/Qwen2.5-7B-Instruct-1M (SIDECAR_INITIAL_MODEL default)
ENGINE_DTYPE:   bfloat16
ENGINE_MAX_MODEL_LEN: 512
ENGINE_GPU_MEMORY_UTILIZATION: 0.70
ENGINE_MAX_PENDING: 10
SIDECAR_L1_CAPACITY_MB: 512
L2_REDIS_MAXMEM: 256mb
Prometheus scrape interval: 15s
```

All benchmarks run against Docker Compose with the standard configuration unless the scenario explicitly overrides a variable.

---

## 3. Benchmark Categories

### 3.1 Load Testing

**Goal:** Determine sustainable throughput under increasing concurrency.

**Endpoint:** `POST /generate` (gateway) and `POST /inference` (engine direct)

**Concurrency sweep:** 1, 2, 4, 8, 16, 32, 64, 128

| Parameter | Values |
|-----------|--------|
| `prompt` | Short (32 tokens), Medium (128 tokens), Long (448 tokens — near `max_model_len`) |
| `max_tokens` | 64, 128, 256 |
| `temperature` | 0.0 (greedy), 0.7 (default), 1.5 (high entropy) |
| `stream` | false |
| `adapter_identifier` | null (base model only) |

**Duration:** 60 seconds per concurrency level, 10-second warm-up excluded from metrics.

**Metrics captured:**
- Requests/sec (client-side)
- `engine_requests_total{status=success|timeout|error}` delta
- `engine_request_duration_seconds` histogram (p50, p95, p99)
- `engine_pending_requests` max observed
- `engine_batch_size` distribution
- HTTP 429 count (queue-full rejections)
- `engine_gpu_memory_used_bytes` peak

**Key questions:**
- At what concurrency does throughput plateau?
- At what concurrency do 429s begin (expected around `ENGINE_MAX_PENDING=10`)?
- How does prompt length affect batch efficiency?

---

### 3.2 Stress Testing

**Goal:** Find the breaking point — where the system degrades, OOMs, or becomes unresponsive.

**Method:** Ramp concurrency from 1 to 256 over 5 minutes (Locust step load), then hold at peak for 2 minutes.

| Scenario | Description |
|----------|-------------|
| **Queue overflow** | Sustained concurrency > `ENGINE_MAX_PENDING` (e.g., 32 concurrent with max_pending=10) |
| **Large output** | `max_tokens=4096` with concurrency=16 |
| **GPU memory pressure** | `ENGINE_GPU_MEMORY_UTILIZATION=0.90` + concurrency=32 |
| **Cold restart under load** | Kill engine mid-run, measure recovery time and request failures |
| **Timeout cascade** | Inject slow prompts (`max_tokens=4096`) while sending fast prompts (`max_tokens=16`) |

**Metrics captured:**
- Error rate over time
- `engine_requests_total{status=timeout}` rate
- `engine_gpu_memory_used_bytes` peak and OOM events
- Gateway 504 (timeout) and 503 (not ready) counts
- System recovery time after overload

---

### 3.3 Latency Profiling

**Goal:** Decompose end-to-end latency into gateway overhead, queue wait, TTFT, and generation time.

**Method:** Single-request httpx client with high-resolution timing (no Locust overhead).

| Segment | How measured |
|---------|-------------|
| **Gateway overhead** | `gateway_total - engine_total` (compare direct engine call vs gateway call) |
| **Queue wait** | `engine_request_duration_seconds - (TTFT + generation_time)` |
| **TTFT** | `engine_time_to_first_token_seconds` |
| **Token generation** | `(tokens_generated - 1) × inter-token latency` |
| **Total** | Client-side wall clock |

**Parameter matrix:**

| prompt_length | max_tokens | temperature | concurrency |
|---------------|------------|-------------|-------------|
| 32 | 64 | 0.0 | 1 |
| 128 | 128 | 0.7 | 1 |
| 448 | 256 | 0.7 | 1 |
| 128 | 128 | 0.7 | 8 |
| 128 | 128 | 0.7 | 16 |

**Repetitions:** 50 per configuration (discard first 5 as warm-up). Report p50, p95, p99, mean, stddev.

---

### 3.4 Throughput Analysis

**Goal:** Measure peak tokens/sec under optimal batching conditions.

**Method:** Saturate the engine at concurrency levels that maximize `engine_tokens_per_second`.

| Configuration | Value |
|---------------|-------|
| Prompt length | 128 tokens (fixed) |
| max_tokens | 256 (fixed) |
| temperature | 0.0 (deterministic) |
| Concurrency sweep | 1, 2, 4, 8, 16, 32 |
| Duration | 120 seconds per level |

**Metrics captured:**
- `engine_tokens_per_second` (gauge, sampled every 1s)
- `engine_tokens_generated_total` delta / wall time
- `engine_batch_size` mean and mode
- Tokens/sec per request vs aggregate tokens/sec
- GPU utilization (pynvml)

**Derived metrics:**
- **Batching efficiency:** `aggregate_tokens_per_sec / (single_request_tokens_per_sec × concurrency)`
- **Optimal concurrency:** concurrency level at which tokens/sec peaks

---

### 3.5 Cache Performance (L1/L2)

**Goal:** Measure KV cache hit rates and latency impact of L1 (DRAM) and L2 (Redis) tiers.

**Scenarios:**

| Scenario | Setup | Expected |
|----------|-------|----------|
| **Cold cache** | Flush L1 and Redis, send unique prompts | 0% hit rate, baseline latency |
| **Warm L1** | Repeat identical prompts within L1 capacity (512 MB) | High L1 hit rate, reduced TTFT |
| **L1 eviction → L2 hit** | Overflow L1 with diverse prompts, then re-request evicted ones | L2 hit, moderate latency improvement |
| **Full miss after L2 eviction** | Overflow both tiers, re-request old prompts | 0% hit rate |
| **Mixed hit ratio** | 70% repeated prompts / 30% unique | Realistic production distribution |

**Parameter matrix:**

| SIDECAR_L1_CAPACITY_MB | L2_REDIS_MAXMEM | Unique prompts | Repeated prompts | Concurrency |
|-------------------------|-----------------|----------------|-------------------|-------------|
| 512 | 256mb | 100 | 0 | 4 |
| 512 | 256mb | 0 | 100 (same prompt) | 4 |
| 512 | 256mb | 50 | 50 | 4 |
| 128 | 64mb | 200 | 100 | 8 |

**Metrics captured:**
- TTFT with cache hit vs miss (delta)
- L1/L2 hit/miss counters (requires prerequisite — see Section 8)
- `engine_time_to_first_token_seconds` distribution per scenario
- Redis `INFO stats` — keyspace hits/misses, used_memory

---

### 3.6 LoRA Adapter Overhead

**Goal:** Quantify the latency and throughput cost of LoRA adapter loading and inference.

**Prerequisites:** `ENGINE_ENABLE_LORA=true`, at least 2 adapters available.

**Scenarios:**

| Scenario | Description |
|----------|-------------|
| **Adapter cold load** | Request with a new adapter — measures fetch + load time |
| **Adapter warm inference** | Request with already-loaded adapter vs base model |
| **Adapter switching** | Alternate between 2 adapters every N requests |
| **Max adapter saturation** | Load `SIDECAR_MAX_ADAPTERS=10` adapters, then request an 11th |
| **Adapter + concurrency** | 8 concurrent requests, 50% base / 50% adapter |

**Metrics captured:**
- `engine_lora_load_duration_seconds` histogram
- `engine_lora_active` gauge over time
- `sidecar_adapter_load_duration_seconds` histogram
- `sidecar_resident_adapters` gauge
- Inference latency delta: adapter vs base model (same prompt/max_tokens)
- `engine_tokens_per_second` with adapter vs without

---

### 3.7 Streaming Performance

**Goal:** Measure token delivery latency and backpressure behavior for SSE streaming responses.

**Method:** httpx async streaming client with per-chunk timestamps.

**Scenarios:**

| Scenario | Parameters |
|----------|------------|
| **Single stream** | 1 client, max_tokens=256, measure inter-token interval |
| **Concurrent streams** | 1, 4, 8, 16 concurrent streaming clients |
| **Stream cancellation** | Client disconnects after receiving 50% of tokens |
| **Mixed stream + non-stream** | 50% streaming / 50% non-streaming at concurrency=8 |

**Metrics captured:**
- **Time-to-first-chunk (TTFC):** Client-side time from request to first SSE data event
- **Inter-token latency (ITL):** Time between consecutive SSE chunks (p50, p95, p99)
- **Total stream duration** vs equivalent non-streaming request duration
- `engine_stream_cancelled_total` counter delta
- `engine_tokens_per_second` during streaming vs non-streaming
- `engine_pending_requests` under concurrent streams

**Derived metrics:**
- **TPOT (Time Per Output Token):** mean ITL across full generation
- **Stream overhead:** `stream_total_duration - nonstream_total_duration` for identical requests

---

### 3.8 Cold Start

**Goal:** Measure time from `docker compose up` to first successful inference response.

**Phases measured:**

| Phase | Start | End | How measured |
|-------|-------|-----|-------------|
| **Sidecar init** | Container start | `GET /health` → 200 | Client polling |
| **Model download** | Sidecar health OK | `GET /ready` → `status=ready` | Client polling + `sidecar_model_load_duration_seconds` |
| **Engine init** | Engine container start | `GET /health` → 200 | Client polling |
| **Engine model load** | Engine detects model in registry | `GET /ready` → 200 | Client polling |
| **First inference** | Send `POST /inference` | Receive 200 response | Client timing |
| **Total cold start** | `docker compose up` | First successful inference | Wall clock |

**Variations:**

| Variable | Values |
|----------|--------|
| Model | Qwen/Qwen2-0.5B (small), Qwen/Qwen2.5-7B-Instruct-1M (default) |
| Model pre-cached | Yes (already on shared volume) / No (full download) |
| `ENGINE_SIDECAR_POLL_INTERVAL` | 1s, 2s (default), 5s |

**Repetitions:** 5 per configuration.

---

### 3.9 Mixed Production Workloads

**Goal:** Simulate realistic traffic patterns combining multiple features simultaneously.

**Workload profiles:**

#### Profile A: Steady-state API traffic
```
Duration: 5 minutes
Concurrency: 8 sustained
Request mix:
  - 60% short prompts (32 tokens), max_tokens=64, stream=false
  - 25% medium prompts (128 tokens), max_tokens=128, stream=true
  - 10% long prompts (448 tokens), max_tokens=256, stream=false
  - 5% invalid requests (empty prompt, max_tokens=0, temperature=3.0)
Adapter usage: none
```

#### Profile B: Adapter-heavy workload
```
Duration: 5 minutes
Concurrency: 8 sustained
Request mix:
  - 40% base model, short prompt, stream=false
  - 30% adapter_A, medium prompt, stream=true
  - 20% adapter_B, medium prompt, stream=false
  - 10% new adapter (cold load during run)
```

#### Profile C: Burst traffic
```
Duration: 3 minutes
Pattern: 10 seconds idle → 30 seconds at concurrency=32 → repeat
Request mix: uniform medium prompts, max_tokens=128
Expected: 429 responses during bursts, clean recovery during idle
```

#### Profile D: Cache exploitation
```
Duration: 5 minutes
Concurrency: 4
Prompt pool: 20 unique prompts
Request pattern: Zipf distribution (80% of requests hit 20% of prompts)
Expected: High cache hit rate after warm-up
```

**Metrics captured (all profiles):**
- All Prometheus engine + sidecar metrics
- Client-side latency percentiles
- Error breakdown (429, 503, 504, 422)
- `engine_tokens_per_second` time series
- `engine_pending_requests` time series
- GPU memory + utilization time series

---

## 4. Metrics Collection Strategy

### 4.1 Client-Side Metrics (Locust / httpx)

Collected by the benchmark harness for every request:

| Metric | Unit | Source |
|--------|------|--------|
| Request latency (wall clock) | ms | httpx response timing |
| TTFC (streaming) | ms | Time to first SSE chunk |
| ITL (streaming) | ms | Inter-chunk intervals |
| HTTP status code | — | Response status |
| Tokens generated | count | `response.tokens_generated` |
| Response text length | chars | `len(response.text)` |

### 4.2 Server-Side Metrics (Prometheus)

Scraped from `GET /metrics` on engine (:8080) and sidecar (:8001):

**Engine metrics:**
- `engine_requests_total` (counter, labels: model, status)
- `engine_request_duration_seconds` (histogram, label: model)
- `engine_time_to_first_token_seconds` (histogram, label: model)
- `engine_tokens_generated_total` (counter, label: model)
- `engine_tokens_per_second` (gauge, label: model)
- `engine_batch_size` (histogram, label: model — buckets: 1, 2, 4, 8, 16, 32)
- `engine_pending_requests` (gauge, label: model)
- `engine_gpu_memory_used_bytes` (gauge, label: device)
- `engine_lora_load_duration_seconds` (histogram, label: adapter)
- `engine_lora_active` (gauge)
- `engine_stream_cancelled_total` (counter, label: model)

**Sidecar metrics:**
- `sidecar_model_load_duration_seconds` (histogram, labels: model, source)
- `sidecar_adapter_load_duration_seconds` (histogram, label: adapter)
- `sidecar_resident_models` (gauge)
- `sidecar_resident_adapters` (gauge)
- `sidecar_download_bytes_total` (counter, label: artifact_type)

### 4.3 System-Level Metrics (pynvml + OS)

Collected in a background thread during all benchmarks:

| Metric | Source | Interval |
|--------|--------|----------|
| GPU utilization % | `pynvml` | 1s |
| GPU memory used/total | `pynvml` | 1s |
| GPU temperature | `pynvml` | 1s |
| CPU utilization | `psutil` | 1s |
| System memory used | `psutil` | 1s |
| Redis memory | `redis-cli INFO memory` | 5s |

---

## 5. SLO Definitions and Compliance

### 5.1 SLO Targets

| SLO | Target | Condition | Metric |
|-----|--------|-----------|--------|
| **TTFT** | < 500 ms (p95) | concurrency ≤ 8, prompt ≤ 128 tokens | `engine_time_to_first_token_seconds` |
| **TPOT** | < 50 ms (p95) | concurrency ≤ 8 | Derived from streaming ITL |
| **E2E latency** | < 5 s (p99) | max_tokens ≤ 256, concurrency ≤ 8 | `engine_request_duration_seconds` |
| **GPU memory** | < 90% utilization | steady state | `engine_gpu_memory_used_bytes / total` |
| **Error rate** | < 1% | excluding 429 (expected under overload) | `engine_requests_total{status=error}` rate |
| **Timeout rate** | < 0.5% | concurrency ≤ ENGINE_MAX_PENDING | `engine_requests_total{status=timeout}` rate |
| **429 rate** | 0% | concurrency ≤ ENGINE_MAX_PENDING | HTTP 429 count |
| **Cold start** | < 120 s | pre-cached model on shared volume | Wall clock to first inference |
| **Adapter load** | < 10 s | adapter already fetched to sidecar | `engine_lora_load_duration_seconds` |

### 5.2 Regression Detection

Each benchmark run produces a JSON results file. The regression checker compares against the most recent baseline:

```
Pass criteria:
  - All SLOs met
  - No metric regressed by > 10% vs baseline (latency, throughput)
  - No new error types observed

Warn criteria:
  - Any metric regressed by 5-10%
  - SLO met but within 20% of threshold

Fail criteria:
  - Any SLO violated
  - Any metric regressed by > 10%
  - New error types or crashes
```

The regression check runs as a post-benchmark script and outputs a pass/warn/fail verdict with details.

---

## 6. Report Generation

### 6.1 JSON Output

Every benchmark run produces a structured JSON file:

```json
{
  "run_id": "bench-20260301-143022",
  "timestamp": "2026-03-01T14:30:22Z",
  "git_sha": "278d972",
  "environment": {
    "gpu": "NVIDIA A100 40GB",
    "model": "Qwen/Qwen2.5-7B-Instruct-1M",
    "engine_max_pending": 10,
    "engine_gpu_memory_utilization": 0.70,
    "engine_max_model_len": 512
  },
  "categories": {
    "load_test": {
      "concurrency_1": { "rps": 12.3, "p50_ms": 82, "p95_ms": 145, "p99_ms": 210, "errors": 0 },
      "concurrency_8": { "rps": 48.1, "p50_ms": 165, "p95_ms": 340, "p99_ms": 520, "errors": 0 }
    },
    "throughput": {
      "peak_tokens_per_sec": 1250,
      "optimal_concurrency": 8,
      "batching_efficiency": 0.82
    },
    "cold_start": {
      "total_seconds": 45.2,
      "phases": { "sidecar_init": 3.1, "model_load": 32.4, "engine_init": 8.2, "first_inference": 1.5 }
    }
  },
  "slo_compliance": {
    "ttft_p95_ms": { "value": 320, "target": 500, "pass": true },
    "tpot_p95_ms": { "value": 38, "target": 50, "pass": true },
    "gpu_utilization_pct": { "value": 78, "target": 90, "pass": true },
    "error_rate_pct": { "value": 0.1, "target": 1.0, "pass": true }
  },
  "regression": {
    "verdict": "pass",
    "baseline_run_id": "bench-20260228-100000",
    "deltas": {
      "load_test.concurrency_8.p95_ms": { "baseline": 330, "current": 340, "delta_pct": 3.0 }
    }
  }
}
```

### 6.2 HTML Report

Generated from JSON using a Jinja2 template:

- **Summary dashboard:** SLO pass/fail badges, overall verdict
- **Latency charts:** Percentile distributions per concurrency level (bar charts)
- **Throughput curves:** Tokens/sec vs concurrency (line chart)
- **Time series:** GPU memory, pending requests, tokens/sec over test duration
- **Comparison table:** Current vs baseline side-by-side
- **Error log:** Any non-200 responses with request details

### 6.3 Output Location

```
benchmarks/
  results/
    bench-{date}-{time}.json
    bench-{date}-{time}.html
  baselines/
    latest.json -> ../results/bench-{date}-{time}.json  (symlink)
```

---

## 7. Implementation Phases

### Phase 1: Harness Foundation
- Set up `benchmarks/` directory structure
- Implement benchmark runner CLI (`python -m benchmarks.run`)
- Build httpx-based client for single-request latency measurement
- Build Locust load profile base classes
- Implement Prometheus metric scraper (query API at run start/end, compute deltas)
- Implement pynvml GPU monitor background thread
- Implement JSON result writer

### Phase 2: Core Benchmarks
- Implement **Load Testing** (3.1) — Locust profiles for concurrency sweep
- Implement **Latency Profiling** (3.3) — httpx precision measurement
- Implement **Throughput Analysis** (3.4) — saturation test
- Implement **Cold Start** (3.8) — Docker Compose lifecycle timing

### Phase 3: Feature Benchmarks
- Implement **Cache Performance** (3.5) — L1/L2 scenarios with cache flush utilities
- Implement **LoRA Adapter Overhead** (3.6) — adapter lifecycle benchmarks
- Implement **Streaming Performance** (3.7) — SSE chunk timing

### Phase 4: Integration Benchmarks
- Implement **Stress Testing** (3.2) — step load and failure scenarios
- Implement **Mixed Production Workloads** (3.9) — all four profiles
- SLO compliance checker
- Regression detection (compare against baselines)

### Phase 5: Reporting and CI
- HTML report generator (Jinja2 template)
- Baseline management (symlinks, history)
- CI integration script (run subset of benchmarks on PR, full suite nightly)
- Documentation and runbook

---

## 8. Known Prerequisites

Issues in the current server code that must be addressed before certain benchmarks can run accurately.

### 8.1 Cache Hit/Miss Metrics

**Status:** Not yet instrumented.

The L1/L2 KV cache does not currently expose hit/miss counters as Prometheus metrics. Cache performance benchmarks (3.5) require:

- `sidecar_cache_l1_hits_total` (counter)
- `sidecar_cache_l1_misses_total` (counter)
- `sidecar_cache_l2_hits_total` (counter)
- `sidecar_cache_l2_misses_total` (counter)
- `sidecar_cache_l1_evictions_total` (counter)

**Workaround:** Until instrumented, cache performance can be inferred indirectly from TTFT deltas between repeated vs unique prompts and Redis `INFO stats` keyspace hits/misses.

### 8.2 Streaming Implementation

**Status:** Verify completeness.

The `stream=true` parameter is accepted by the API, and `engine_stream_cancelled_total` exists as a metric. Verify that:
- Engine returns proper SSE (`text/event-stream`) responses
- Each token is flushed as an individual SSE event
- Client disconnect triggers cancellation and metric increment
- Streaming works through the gateway (proxy pass-through)

Streaming benchmarks (3.7) depend on these working correctly.

### 8.3 Gateway Metrics

**Status:** No gateway-specific Prometheus metrics exist.

Gateway overhead measurement (3.3) currently relies on comparing direct-engine vs gateway-routed latency. For finer granularity, consider adding:

- `gateway_requests_total` (counter, labels: model, status)
- `gateway_request_duration_seconds` (histogram)
- `gateway_upstream_duration_seconds` (histogram — time spent waiting on engine)

### 8.4 Batch Size Observability

**Status:** `engine_batch_size` histogram exists but verify it is recorded on every batch cycle.

Throughput analysis (3.4) and load testing (3.1) depend on accurate batch size reporting to correlate batching efficiency with throughput.

### 8.5 Engine Queue Wait Time

**Status:** Not directly exposed.

Latency profiling (3.3) decomposes queue wait as `request_duration - (TTFT + generation)`. A dedicated metric would improve accuracy:

- `engine_queue_wait_seconds` (histogram)

---

## Appendix: Endpoint Reference

| Service | Port | Endpoint | Method | Benchmark relevance |
|---------|------|----------|--------|---------------------|
| Gateway | 8000 | `/health` | GET | Cold start timing |
| Gateway | 8000 | `/generate` | POST | All load/latency benchmarks |
| Engine | 8080 | `/health` | GET | Cold start timing |
| Engine | 8080 | `/ready` | GET | Cold start timing |
| Engine | 8080 | `/inference` | POST | Direct engine benchmarks |
| Engine | 8080 | `/metrics` | GET | All benchmarks (Prometheus) |
| Sidecar | 8001 | `/health` | GET | Cold start timing |
| Sidecar | 8001 | `/ready` | GET | Cold start timing |
| Sidecar | 8001 | `/load/{model}` | POST | Cold start, model load benchmarks |
| Sidecar | 8001 | `/unload/{model}` | POST | Stress test recovery |
| Sidecar | 8001 | `/registry/models` | GET | Cold start polling verification |
| Sidecar | 8001 | `/registry/adapters` | GET | Adapter benchmarks |
| Sidecar | 8001 | `/adapter/fetch/{id}` | POST | Adapter cold load benchmarks |
| Sidecar | 8001 | `/metrics` | GET | All benchmarks (Prometheus) |

## Appendix: Configuration Knobs Exercised

| Env Var | Default | Benchmarks that vary it |
|---------|---------|------------------------|
| `ENGINE_MAX_PENDING` | 10 | 3.1, 3.2, 3.9C |
| `ENGINE_GPU_MEMORY_UTILIZATION` | 0.70 | 3.2 |
| `ENGINE_MAX_MODEL_LEN` | 512 | 3.1, 3.3 (prompt length ceiling) |
| `ENGINE_ENABLE_LORA` | false | 3.6, 3.9B |
| `ENGINE_SIDECAR_POLL_INTERVAL` | 2.0 | 3.8 |
| `SIDECAR_L1_CAPACITY_MB` | 512 | 3.5 |
| `SIDECAR_MAX_ADAPTERS` | 10 | 3.6 |
| `L2_REDIS_MAXMEM` | 256mb | 3.5 |
| `GATEWAY_REQUEST_TIMEOUT` | 300.0 | 3.2 |
