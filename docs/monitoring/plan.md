# Monitoring System for Inference Server

## Context

The inference server already exposes `/metrics` with Prometheus-formatted metrics, but many are **defined but never populated** (TTFT, tokens_per_second, pending_requests, gpu_memory_used_bytes). Token counting uses `split()` (word count, not token count). There is no per-request session tracking, no GPU monitoring (no pynvml), and no aggregated summary endpoint. The goal is to add a `/metrics_summary` endpoint that provides actionable, per-session aggregated metrics demonstrating deep understanding of LLM inference systems.

---

## Metrics Design

### Your proposed metrics + what's missing

Your 5 proposed metrics (avg TTFT, inter-token latency, generation length, GPU util, GPU memory) are a good start but only cover the "happy path latency" view. To demonstrate production-grade inference knowledge, add:

**Latency decomposition** (shows you understand the prefill/decode pipeline):
- **Queue wait time** per model (time from request arrival to processing start)
- **Prefill latency** per model (compute-only, distinct from TTFT which includes queue wait)
- **End-to-end latency percentiles** (p50, p95, p99 -- not just averages)

**Throughput & efficiency:**
- **Input tokens per request** (histogram) -- determines prefill cost, KV memory
- **Output tokens per request** (histogram) -- fix current `split()` bug, use tokenizer
- **Tokens/second throughput** (input+output, system-wide)
- **Requests/second**
- **Batch size utilization** (already defined, never observed -- shows continuous batching understanding)

**KV cache** (your architecture's unique value):
- **Cache hit rate** (already computed in KVBlockRegistry.stats(), just not exposed)
- **L1 utilization** (already instrumented)
- **Eviction rate**

**Reliability/SLA:**
- **Error rate** by type (success/error/timeout -- already tracked via counter labels)
- **Queue depth** (current + max over session)

**LoRA-specific:**
- **Adapter swap latency** (already instrumented)
- **Active adapter count**
- **Per-adapter request distribution**

**GPU (via pynvml):**
- **Compute utilization %** (not just memory -- low compute + high memory = memory-bound)
- **Memory breakdown** (allocated vs total)
- **Power draw** (tokens-per-watt efficiency)

---

## Architecture Decision

**Recommended: Hybrid in-process collector + storage abstraction (Option D)**

Rejected alternatives:
- **Separate monitoring microservice** -- reinvents Prometheus, overengineered for current stage
- **Thin API over PromQL** -- great for operational dashboards but wrong for per-session/per-request detail

The hybrid approach:
1. Each engine instance has an **in-process `SessionCollector`** with a bounded ring buffer
2. `/metrics_summary` reads from both the Prometheus registry (aggregate stats) and the SessionCollector (per-request detail)
3. A **`StorageBackend` ABC** abstracts persistence (local JSON now, MongoDB later)
4. For multi-node: gateway fans out to all engine `/metrics_summary` endpoints

This matches industry patterns (vLLM, TGI, TensorRT-LLM all expose metrics in-process) while providing a clean migration path.

---

## Implementation Plan

### Decisions
- **Token counting**: Use vLLM's `output.outputs[0].token_ids` length (zero overhead). Mock engine uses `len(text) // 4` estimate. For input tokens, prefer `len(output.prompt_token_ids)` if available on `RequestOutput`; fall back to `len(self.engine.get_tokenizer().encode(prompt))`.
- **Timing data flow**: Shared `request_timing: Dict[str, TimingInfo]` dict parallel to `request_futures`. Cleaned up when request completes.
- **LoRA return type**: `ensure_adapter_loaded()` returns `tuple[LoRARequest, float]` where the float is the swap duration in seconds (0.0 on cache hit). This is cleaner than mutating `LoRARequest` and makes the API explicit. All callers (`engine.py`, `mock_engine.py`) must destructure accordingly.
- **Prometheus from shared/**: `shared/monitoring/gpu.py` must NOT import from `engine/metrics.py` (would create shared→engine circular dependency). Instead, `GPUMonitor` accepts an optional `on_sample` callback in `__init__()`. The caller in `api.py` passes a callback that updates the Prometheus gauges.
- **Mock engine timing**: Record timing in `step()` only (not `_resolve_pending()`). `step()` is what `continuous_batching_loop()` calls and is the authoritative processing path. `_resolve_pending()` is a convenience shortcut that should delegate to `step()`.
- **Graceful degradation**: All external data sources (sidecar KV cache, GPU monitor) return empty dicts `{}` on failure. The `/metrics_summary` endpoint must never fail due to an unavailable subsystem.
- **Scope**: All 3 phases (core + GPU + persistence) in one go.

### Phase 1: Core instrumentation + SessionCollector integration (8 files)

> **Status:** `shared/monitoring/` module (models.py, collector.py, gpu.py, storage.py) is **already implemented**.
> The work in this phase is entirely about **wiring the existing monitoring infrastructure into the engine**.

#### 1a. Instantiate SessionCollector in `engine/api.py` *(prerequisite for all metrics)*

- Add `from shared.monitoring import SessionCollector, TimingInfo, RequestRecord` to imports
- Create module-level `_collector: Optional[SessionCollector] = None`
- Instantiate `_collector = SessionCollector()` in `lifespan()` startup (alongside `_engine`)
- Pass `_collector` reference to both `Engine` and `MockLLMEngine` (or keep in api.py and feed records after each request)

#### 1b. Instrument per-request timing in `engine/engine.py` *(latency metrics)*

Source: real vLLM engine. Key locations: `add_request()` (line ~98), `continuous_batching_loop()` (line ~139-149).

- Add `self.request_timings: Dict[str, TimingInfo]` dict to `Engine.__init__()` (parallel to `request_futures`)
- Accept `collector: SessionCollector` in `Engine.__init__()` and store as `self.collector`
- In `add_request()`: create `TimingInfo(submitted_at=time.time(), adapter_id=adapter_identifier)`, store in `self.request_timings[request_id]`
- In `add_request()`: destructure LoRA result as `lora_request, swap_duration = await self.lora_manager.ensure_adapter_loaded(...)`, store `timing.adapter_swap_latency_s = swap_duration`
- In `continuous_batching_loop()`, after `outputs_list = await asyncio.to_thread(self.engine.step)`:
  - For each output, if `timing.processing_started_at == 0.0`: set `timing.processing_started_at = time.time()` (first time request_id appears in step output = processing start proxy)
  - If `timing.first_token_at == 0.0` and output has generated text: set `timing.first_token_at = time.time()`
  - On every step: set `timing.last_step_at = time.time()`, increment `timing.step_count`
  - When `output.finished`: set `timing.finished_at = time.time()`, `timing.output_tokens = timing.step_count`. For input tokens: use `len(output.prompt_token_ids)` if available on vLLM's `RequestOutput`, otherwise fall back to `timing.input_tokens` (pre-set in `add_request()` via tokenizer)
  - Build `RequestRecord.from_timing(request_id, model_id, timing, status="success")`, call `self.collector.record_request(record)`
  - Clean up `self.request_timings[request_id]`
- Add `tokenize(text)` method: `self.engine.get_tokenizer().encode(text)` for accurate token counting

#### 1c. Mirror timing instrumentation in `engine/mock_engine.py`

Same structure as 1b but for MockLLMEngine:
- Add `self.request_timings: Dict[str, TimingInfo]` to `__init__()` (line ~47)
- Accept `collector: SessionCollector` in `__init__()` and store as `self.collector`
- In `add_request()` (line ~82): create `TimingInfo(submitted_at=time.time(), adapter_id=adapter_identifier)`
- In `add_request()`: destructure LoRA result as `lora_request, swap_duration = await self.lora_manager.ensure_adapter_loaded(...)`, store `timing.adapter_swap_latency_s = swap_duration`
- In `step()` only (NOT `_resolve_pending()`): set all timing fields, build `RequestRecord`, call `self.collector.record_request()`. `_resolve_pending()` is a convenience that calls `step()` internally, so timing is recorded once through the `step()` path
- Add `tokenize(text)` stub: `len(text) // 4` estimate

#### 1d. Fix token counting bug + wire collector in `engine/api.py`

Current bug: `api.py:206` uses `len(result.split())` (word count, not token count).

- Replace `len(result.split())` with `len(_engine.tokenize(result))` for accurate output token count
- Compute `input_tokens = len(_engine.tokenize(request.prompt))` for input token count
- In the success path (after line ~203): if collector wiring is in api.py rather than in the engine loop, build `RequestRecord` and call `_collector.record_request(record)`
- In the `except asyncio.TimeoutError` block (line ~216): record with `status="timeout"`
- In the `except Exception` block (line ~222): record with `status="error"`
- Generate a `request_id` (e.g., `str(uuid4())`) at the top of `inference()` for tracking

#### 1e. Add queue depth tracking *(reliability metrics)*

- In `SessionCollector` (`collector.py`): add `_current_queue_depth`, `_max_queue_depth` fields to `__init__()`, add `inc_queue_depth()` and `dec_queue_depth()` methods (thread-safe), expose in `get_summary()` under `"queue"` key
- In `api.py` `inference()`: call `_collector.inc_queue_depth()` at request entry, `_collector.dec_queue_depth()` in a `finally` block
- Populate existing Prometheus gauge `engine_pending_requests`: `.inc()` at request entry, `.dec()` in `finally`
- Note: `model_id = _config.model_name` must move before the `try` block so `finally` can reference it

#### 1f. Wire batch size tracking *(throughput metrics)*

- In `engine.py` `continuous_batching_loop()`, after `outputs_list = await asyncio.to_thread(self.engine.step)`:
  - Call `self.collector.record_batch_size(len(outputs_list))`
  - Call `metrics.engine_batch_size.labels(model=self.config.model_name).observe(len(outputs_list))`
- Same in `mock_engine.py` after `await self.step()`
- `SessionCollector.record_batch_size()` already exists (collector.py:33) -- just needs to be called

#### 1g. Wire LoRA metrics

**Adapter swap latency:**
- Add `adapter_swap_latency_s: float = 0.0` field to `TimingInfo` and `RequestRecord` in `models.py`, wire through `from_timing()`
- Modify `LoRAManager.ensure_adapter_loaded()` in `lora_manager.py` to return `tuple[LoRARequest, float]` instead of just `LoRARequest`. The float is the swap duration in seconds. On cache hit (fast path, line 96), return `(lora_request, 0.0)`. On GPU load (line 162-163), return `(lora_request, duration)`. The duration is already computed at line 163 (`duration = time.time() - start`), just not returned
- Callers in `Engine.add_request()` and `MockLLMEngine.add_request()` must destructure: `lora_request, swap_duration = await self.lora_manager.ensure_adapter_loaded(...)`
- Store swap duration in `TimingInfo.adapter_swap_latency_s`
- Update `_lora_summary()` in `collector.py` to compute `swap_count` and `avg_swap_latency_ms` from records where `adapter_swap_latency_s > 0`

**Active adapter count (live GPU state):**
- Add `loaded_count` property to `LoRAManager`: `return len(self._loaded)`
- Add `loaded_keys` property to `LoRAManager`: `return list(self._loaded.keys())`
- Add optional `lora_state_fn: Optional[Callable[[], dict]]` callback to `SessionCollector.__init__()` -- callable returns `{"loaded_count": int, "loaded_keys": list}`
- In `get_summary()`, call `lora_state_fn()` if set, merge into `lora` section as `gpu_loaded_count` and `gpu_loaded_adapters`
- When constructing SessionCollector in `api.py`, pass lambda reading from `_engine.lora_manager.loaded_count` / `loaded_keys`

**Per-adapter request distribution:**
- Add `engine_lora_requests_total` Counter (labels: `adapter`) to `metrics.py`
- Increment in `api.py` after successful inference when `adapter_identifier` is not None
- Update `_lora_summary()` in `collector.py` to return `per_adapter_requests` dict (Counter of adapter_id values from records)

#### 1h. Add new Prometheus metrics to `engine/metrics.py`

New histograms:
- `engine_queue_wait_seconds` (Histogram, labels: model)
- `engine_prefill_seconds` (Histogram, labels: model)
- `engine_inter_token_latency_seconds` (Histogram, labels: model)
- `engine_input_tokens_per_request` (Histogram, labels: model, buckets: [16,32,64,128,256,512,1024,2048,4096])
- `engine_output_tokens_per_request` (Histogram, labels: model, buckets: [16,32,64,128,256,512,1024,2048])
- `engine_decode_tokens_per_second` (Histogram, labels: model)

New counters:
- `engine_lora_requests_total` (Counter, labels: adapter)

Populate existing unpopulated metrics from timing data:
- `engine_time_to_first_token_seconds` -- observe `record.ttft_s`
- `engine_tokens_per_second` -- set from `record.tokens_per_second` or from collector throughput
- `engine_pending_requests` -- inc/dec in api.py (see 1e)
- `engine_batch_size` -- observe in batching loop (see 1f)

Observe all new histograms in the engine batching loop (or api.py) after building each `RequestRecord`.

#### 1i. Wire KV cache metrics via sidecar HTTP bridge

The KV cache data lives in the **sidecar process** (separate container). The engine already calls `{sidecar_url}/registry/models` during startup, so the HTTP bridge pattern is established.

**Sidecar side** (2 files):
- Extend `KVBlockRegistry.stats()` in `sidecar/kv_block_registry.py` to include `l1_utilization_ratio` and `eviction_rate` (add `_eviction_window_start` timestamp, compute `evictions / elapsed_seconds`)
- Extend `GET /cache/stats` endpoint in `sidecar/api.py` to also return `l1_capacity_bytes` and `l1_utilization_ratio` (read from Prometheus gauge `l1_cache_utilization_ratio._value.get()` or from `MultiTieredCacheManager` stats)

**Engine side:**
- In `/metrics_summary` handler, make `httpx.get(f"{_config.sidecar_url}/cache/stats")` call wrapped in try/except
- **Graceful degradation**: if the sidecar is unreachable (connection refused, timeout, non-200), log a warning and use `kv_cache = {}`. The endpoint must never fail because the sidecar is down
- Use a short timeout (2-3s) for the sidecar call to avoid blocking the summary response
- Merge result into the `kv_cache` section of the summary response (caller merges into the dict returned by `get_summary()`, don't complicate collector's API)

#### 1j. Add `GET /metrics_summary` endpoint to `engine/api.py`

Endpoint that orchestrates all data sources:
1. Call `_collector.get_summary(model_id=_config.model_name)` for per-request aggregations
2. Call `_gpu_monitor.get_summary()` (if available) and merge into `gpu` section
3. Call `GET {sidecar_url}/cache/stats` (if reachable) and merge into `kv_cache` section
4. Return merged JSON

Returns JSON:
```json
{
  "session": { "start_time", "uptime_seconds", "total_requests", "requests_per_second" },
  "latency": {
    "ttft": { "avg", "p50", "p95", "p99" },
    "prefill": { "avg", "p50", "p95", "p99" },
    "queue_wait": { "avg", "p50", "p95", "p99" },
    "inter_token": { "avg", "p50", "p95", "p99" },
    "e2e": { "avg", "p50", "p95", "p99" }
  },
  "throughput": { "requests_per_second", "input_tokens_per_second", "output_tokens_per_second" },
  "generation": {
    "input_length": { "min", "max", "mean" },
    "output_length": { "min", "max", "mean" }
  },
  "batching": { "avg_batch_size", "max_batch_size" },
  "queue": { "current_depth", "max_depth" },
  "errors": { "error_rate", "timeout_rate", "total_errors" },
  "lora": { "gpu_loaded_count", "gpu_loaded_adapters", "swap_count", "avg_swap_latency_ms", "per_adapter_requests" },
  "gpu": { "current": { "compute_utilization_pct", "memory_used_bytes", "memory_total_bytes", "memory_utilization_pct", "power_watts" }, "session": { "avg_compute_utilization_pct", "max_compute_utilization_pct", "max_memory_used_bytes", "avg_power_watts", "max_power_watts" }, "tokens_per_watt": 0.0 },
  "kv_cache": { "hit_rate", "l1_utilization_ratio", "l1_capacity_bytes", "l1_used_bytes", "eviction_count", "eviction_rate" }
}
```

#### Phase 1 Dependency Graph

```
1a (instantiate collector) ──────────────────────────────────────┐
  │                                                               │
1b (engine.py timing)────┐                                       │
1c (mock_engine.py timing)┤                                      │
  │                       ├─→ 1d (api.py wiring + token fix)     │
  │                       │     │                                 │
  │                       │   1e (queue depth)                    │
  │                       │   1f (batch size)                     │
  │                       │   1g (LoRA metrics)                   │
  │                       │     │                                 │
  │                       └─→ 1h (Prometheus metrics) ────────────┤
  │                                                               │
1i (KV cache sidecar bridge) ────────────────────────────────────┤
  │                                                               │
  └──────────────────────────────────────────────────────────→ 1j (/metrics_summary endpoint)
```

**Execution order:** 1a → (1b + 1c parallel) → 1d → (1e + 1f + 1g + 1h parallel) → 1i → 1j

---

### Phase 2: GPU telemetry integration (4 files)

> **Status:** `shared/monitoring/gpu.py` (`GPUMonitor`) is **already implemented** with pynvml polling, graceful degradation, and `get_summary()`.
> The work in this phase is about **instantiating, configuring, and connecting GPUMonitor to the engine lifecycle and Prometheus**.

#### 2a. Add GPU monitoring config to `engine/config.py` and `server_config.yaml`

- Add fields to `EngineConfig` in `config.py`: `gpu_monitor_enabled: bool = True`, `gpu_poll_interval: float = 2.0`, `gpu_device_index: int = 0`
- Add corresponding entries under `engine:` section in `server_config.yaml`

#### 2b. Add new Prometheus gauges to `engine/metrics.py`

- `engine_gpu_compute_utilization_percent` (Gauge, labels: device)
- `engine_gpu_memory_total_bytes` (Gauge, labels: device) -- companion to existing `engine_gpu_memory_used_bytes`
- `engine_gpu_power_watts` (Gauge, labels: device)

#### 2c. Add power session aggregates + Prometheus callback to `shared/monitoring/gpu.py`

- Add `_sum_power`, `_max_power` accumulators to `GPUMonitor.__init__()`
- Add optional `on_sample: Optional[Callable[[dict], None]]` callback parameter to `GPUMonitor.__init__()`. This callback receives the current snapshot dict after each poll
- Update `_sample()` to accumulate power data, then call `self._on_sample(snapshot)` if set
- Expose `avg_power_watts` and `max_power_watts` in `get_summary()["session"]`
- **Do NOT import from `engine/metrics.py`** — this would create a shared→engine circular dependency. The Prometheus gauge updates happen in the callback provided by `api.py`:
  ```python
  def _gpu_on_sample(snapshot: dict):
      metrics.engine_gpu_compute_utilization_percent.labels(device="0").set(snapshot["compute_utilization_pct"])
      metrics.engine_gpu_memory_used_bytes.labels(device="0").set(snapshot["memory_used_bytes"])
      metrics.engine_gpu_memory_total_bytes.labels(device="0").set(snapshot["memory_total_bytes"])
      metrics.engine_gpu_power_watts.labels(device="0").set(snapshot["power_watts"])
  ```

#### 2d. Instantiate GPUMonitor in `engine/api.py` lifespan

- Add module-level `_gpu_monitor: Optional[GPUMonitor] = None`
- In `lifespan()` startup: if `_config.gpu_monitor_enabled`, create `GPUMonitor(poll_interval=_config.gpu_poll_interval, device_index=_config.gpu_device_index)` and call `await _gpu_monitor.start()`
- In `lifespan()` shutdown: call `await _gpu_monitor.stop()`

#### 2e. Wire GPU data into `/metrics_summary` endpoint

- In the `/metrics_summary` handler (created in 1j): call `_gpu_monitor.get_summary()` and merge into the `gpu` section
- Compute derived `tokens_per_watt`: `collector_summary["throughput"]["output_tokens_per_second"] / gpu_summary["current"]["power_watts"]` (guard against zero)
- Modify `SessionCollector.get_summary()` to accept optional `gpu_summary: dict` parameter (or let the caller merge -- simpler)

#### Phase 2 Dependency Graph

```
2a (config) ──────────┐
2b (Prometheus gauges) ┤
2c (gpu.py power agg.) ┤
                       ├─→ 2d (instantiate in lifespan) ─→ 2e (wire into /metrics_summary)
                       │                                         ↑
                       │                                    (depends on 1j)
```

**Execution order:** (2a + 2b + 2c parallel) → 2d → 2e (after Phase 1's 1j is complete)

### Phase 3: Storage persistence (~100 lines, 2 files)

- `shared/monitoring/storage.py`:
  - `MetricsStorageBackend` ABC with `store_records()` and `query_records()`
  - `LocalJSONLStore` -- append-only JSON lines file
- Background flush task in engine lifespan (configurable interval, default 30s)
- Add `monitoring:` section to `server_config.yaml`:
  ```yaml
  monitoring:
    storage_backend: "local"  # or "mongodb"
    local_store_path: "/mnt/models/metrics.jsonl"
    buffer_size: 10000
    flush_interval: 30
  ```

### Phase 4: MongoDB migration (future, ~60 lines)

- `MongoDBStore(MetricsStorageBackend)` using `motor` async driver
- Config-driven backend selection -- one-line change in config
- Add `motor` to optional dependencies

### Phase 5: Multi-node aggregation (future, ~80 lines)

- Gateway `/cluster/metrics_summary` fans out to all engines via `MODEL_SERVICE_MAP`
- Merges per-node summaries
- Uncomment Prometheus in docker-compose

### Phase 6: Distributed tracing with OpenTelemetry (future, ~200 lines)

Per-request tracing to decompose latency across the full inference pipeline. Each request gets a trace ID; every stage (queuing, routing, tokenization, prefill, decode, etc.) becomes a span with precise timing. GPU memory snapshots are attached as span attributes so you can correlate memory pressure with specific requests.

#### 6a. Add OpenTelemetry dependencies

- Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` to requirements
- Optional: `opentelemetry-instrumentation-fastapi` for automatic HTTP span creation on incoming requests

#### 6b. Initialize OTel tracer + provider in `engine/api.py` lifespan

- Create a `TracerProvider` with a `BatchSpanProcessor` exporting to an OTLP endpoint (configurable via `server_config.yaml`)
- Register it as the global tracer provider
- Create a module-level `tracer = trace.get_tracer("inference-server")`
- Add config fields to `engine/config.py`: `otel_enabled: bool = False`, `otel_endpoint: str = "http://localhost:4317"`, `otel_service_name: str = "inference-engine"`
- Add corresponding `tracing:` section to `server_config.yaml`:
  ```yaml
  tracing:
    enabled: false
    otel_endpoint: "http://localhost:4317"
    service_name: "inference-engine"
  ```
- On shutdown: call `tracer_provider.shutdown()` to flush pending spans

#### 6c. Instrument the request lifecycle with spans

Create a root span per request in `api.py` `inference()`, then child spans for each pipeline stage. The trace hierarchy:

```
inference_request (root span)
├── queue_wait          — from request arrival to processing start
├── adapter_routing     — LoRA adapter resolution + swap (if applicable)
├── tokenization        — input prompt tokenization
├── prefill             — first forward pass (prompt encoding → first token)
├── decode              — autoregressive token generation loop
│   ├── decode_step_1
│   ├── decode_step_2
│   └── ...
└── detokenization      — token IDs → output text
```

**In `engine/api.py`:**
- Start root span `inference_request` at the top of `inference()`. Set attributes: `request.id`, `request.model`, `request.prompt_length`, `request.max_tokens`
- Start child span `queue_wait` immediately. End it when the engine begins processing (signaled via timing info or a callback)
- Propagate the span context into `Engine.add_request()` so child spans can be created inside the engine

**In `engine/engine.py` (and `mock_engine.py`):**
- Accept an optional `trace.Context` in `add_request()` and store it alongside `request_timings`
- `adapter_routing` span: wrap `ensure_adapter_loaded()` call. Set attributes: `adapter.id`, `adapter.cache_hit` (bool), `adapter.swap_duration_ms`
- `tokenization` span: wrap `tokenize(prompt)` call. Set attribute: `tokens.input_count`
- `prefill` span: start when request first appears in `step()` output, end when first token is produced (i.e., `first_token_at` is set). Set attribute: `prefill.latency_ms`
- `decode` span: start after first token, end when `output.finished`. Set attributes: `tokens.output_count`, `decode.tokens_per_second`, `decode.step_count`. Optionally create a child span per decode step (disable by default as it's high-volume — controlled via `tracing.per_step_spans: false` config)
- `detokenization` span: wrap the final token-to-text conversion if it's a distinct step

#### 6d. Attach GPU memory snapshots to spans

- At the start and end of each major span (`prefill`, `decode`), read current GPU memory from `GPUMonitor` and attach as span attributes:
  - `gpu.memory_used_bytes_start`, `gpu.memory_used_bytes_end`
  - `gpu.memory_delta_bytes` (end − start, shows per-stage memory impact)
  - `gpu.compute_utilization_pct` (snapshot at span start)
- This lets you answer questions like: "Which stage is consuming the most GPU memory?" or "Is the prefill stage memory-bound?"
- Guard with `if _gpu_monitor is not None` — tracing should work without GPU monitoring

#### 6e. Inject trace context into logs

- Configure the OTel logging integration so that all Python `logging` calls automatically include `trace_id` and `span_id` fields
- Use `opentelemetry.instrumentation.logging` or manually add a logging filter that reads from `trace.get_current_span().get_span_context()`
- Log format becomes: `%(asctime)s [%(trace_id)s/%(span_id)s] %(levelname)s %(message)s`
- This enables querying logs by trace ID in Loki/Grafana/any log aggregator

#### 6f. Propagate trace context across the sidecar boundary

- When the engine calls the sidecar's HTTP endpoints (e.g., `/cache/stats`, `/registry/models`), inject the W3C `traceparent` header using OTel's `inject()` propagator
- In the sidecar's FastAPI app, extract the trace context from incoming headers and create child spans for KV cache operations
- This gives end-to-end visibility when a request touches both the engine and sidecar processes

#### Phase 6 Dependency Graph

```
6a (dependencies) ─→ 6b (tracer init + config) ─┐
                                                  ├─→ 6c (span instrumentation) ─→ 6d (GPU attrs)
                                                  ├─→ 6e (log correlation)
                                                  └─→ 6f (sidecar propagation)
```

**Execution order:** 6a → 6b → (6c + 6e + 6f parallel) → 6d (after 6c, depends on span structure)

#### Recommended backend stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Trace collector | **OpenTelemetry Collector** (sidecar or daemonset) | Receives OTLP spans, fans out to backends |
| Trace storage | **Jaeger** or **Grafana Tempo** | Store + query traces by ID, service, duration |
| Log aggregation | **Grafana Loki** | Query logs by trace_id label |
| Visualization | **Grafana** | Unified dashboards: traces (Tempo) + metrics (Prometheus) + logs (Loki) |

#### Example trace output

For a single request, the trace waterfall in Jaeger/Tempo would look like:

```
[inference_request]  ──────────────────────────────────── 320ms
  [queue_wait]       ████                                  45ms
  [adapter_routing]      ██                                12ms  (adapter=llama-lora, cache_hit=true)
  [tokenization]           █                                3ms  (input_tokens=128)
  [prefill]                 ████████                       80ms  (gpu_mem_delta=+256MB)
  [decode]                          ████████████████████  180ms  (output_tokens=64, 355 tok/s)
```

---

## Files to modify/create

| File | Status | Action |
|------|--------|--------|
| `shared/monitoring/__init__.py` | **EXISTS** | No changes needed |
| `shared/monitoring/models.py` | **EXISTS** | Modify -- add `adapter_swap_latency_s` field to TimingInfo + RequestRecord |
| `shared/monitoring/collector.py` | **EXISTS** | Modify -- add queue depth tracking (inc/dec/max), `lora_state_fn` callback, update `_lora_summary()` for swap latency + per-adapter breakdown + per_adapter_requests |
| `shared/monitoring/gpu.py` | **EXISTS** | Modify -- add power session aggregates, add `on_sample` callback (NOT direct Prometheus import) |
| `shared/monitoring/storage.py` | **EXISTS** | No changes needed |
| `data_plane/inference/engine/metrics.py` | Modify | Add 6 new histograms (queue_wait, prefill, inter_token, input_tokens, output_tokens, decode_tps), 1 counter (lora_requests), 3 gauges (gpu compute, gpu memory total, gpu power) |
| `data_plane/inference/engine/engine.py` | Modify | Add `request_timings` dict, full lifecycle timing in `add_request()` + `continuous_batching_loop()`, batch size recording, `tokenize()` method |
| `data_plane/inference/engine/mock_engine.py` | Modify | Mirror timing instrumentation from engine.py, add `tokenize()` stub |
| `data_plane/inference/engine/api.py` | Modify | Instantiate SessionCollector + GPUMonitor, fix token counting bug, wire collector for all request outcomes, add queue depth tracking, add `GET /metrics_summary` endpoint |
| `data_plane/inference/engine/config.py` | Modify | Add `gpu_monitor_enabled`, `gpu_poll_interval`, `gpu_device_index` fields |
| `data_plane/inference/engine/lora_manager.py` | Modify | Return `tuple[LoRARequest, float]` from `ensure_adapter_loaded()`, add `loaded_count` and `loaded_keys` properties |
| `data_plane/inference/sidecar/kv_block_registry.py` | Modify | Extend `stats()` with `l1_utilization_ratio` and `eviction_rate` |
| `data_plane/inference/sidecar/api.py` | Modify | Extend `GET /cache/stats` response with L1 utilization + eviction rate |
| `server_config.yaml` | Modify | Add GPU monitoring config under `engine:` section |
| `shared/monitoring/tracing.py` | **CREATE** (Phase 6) | OTel tracer initialization, provider setup, helper to attach GPU attributes to spans |
| `data_plane/inference/engine/config.py` | Modify (Phase 6) | Add `otel_enabled`, `otel_endpoint`, `otel_service_name` fields |
| `data_plane/inference/engine/api.py` | Modify (Phase 6) | Init tracer provider in lifespan, root span per request, queue_wait span |
| `data_plane/inference/engine/engine.py` | Modify (Phase 6) | Accept trace context in `add_request()`, create adapter_routing/tokenization/prefill/decode spans |
| `data_plane/inference/engine/mock_engine.py` | Modify (Phase 6) | Mirror span instrumentation from engine.py |
| `data_plane/inference/sidecar/api.py` | Modify (Phase 6) | Extract W3C traceparent from incoming requests, create child spans for KV cache ops |
| `server_config.yaml` | Modify (Phase 6) | Add `tracing:` section |
| `requirements.txt` | Modify (Phase 6) | Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` |

## Verification

1. **Unit tests**: Test SessionCollector aggregation logic (min/max/mean/percentiles)
2. **Mock engine test**: Start server with mock engine, send requests, hit `/metrics_summary`, verify all fields populated
3. **Integration**: `curl localhost:8080/metrics_summary | python -m json.tool` -- verify JSON structure
4. **GPU test**: On GPU machine, verify pynvml metrics appear; on dev machine, verify graceful fallback
5. **Persistence**: Check that `metrics.jsonl` is written after flush interval
6. **Prometheus**: Verify new histograms appear in `/metrics` output
7. **Tracing (Phase 6)**: Send a request, verify trace appears in Jaeger/Tempo with all expected spans (queue_wait, adapter_routing, tokenization, prefill, decode). Verify `trace_id` appears in log output. Verify GPU memory attributes are present on prefill/decode spans
