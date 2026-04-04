# Portfolio Roadmap — Inference Engineering

## Context

The inference server (gateway, vLLM engine, sidecar, KV offload) is functional. The goal now is to maximize portfolio and interview value for inference engineering roles, not ship a product nobody will use.

---

## Phase 0 — Complete the single-GPU server

Finish the two incomplete features that directly matter for benchmarking and architectural completeness. Everything else in the robustness plan is deferred to Phase 5.

### L1 KV cache engine-side integration
The sidecar-side L1 cache (allocation, LRU eviction, gRPC servicer) is implemented. What's missing is the engine-side offload hooks — the code in `data_plane/inference/engine/kv_offload/` that actually calls the sidecar's gRPC `StoreBlock`/`LoadBlock` during inference. Without this, the prefix caching benchmarks only measure vLLM's built-in cache, not the custom offload layer.

- Wire `sidecar_handler.py` and `sidecar_backend.py` into the vLLM engine path
- Enable via `ENGINE_KV_OFFLOAD=true` config flag (already exists)
- Validate with a round-trip test: store block via gRPC, load it back, verify data integrity

### Graceful shutdown with request draining
Current shutdown cancels in-flight tasks immediately. Add drain logic:

- On SIGTERM, stop accepting new requests (return 503)
- Wait for in-flight requests to complete (configurable `drain_timeout`, default 30s)
- Then cancel remaining work and shut down

Small effort, completes what's already partially there in the engine lifespan.

---

## Phase 1 — Finish Benchmarking

Complete all five existing experiments and produce polished results:

1. **Latency decomposition** — prefill vs decode across SISO/SILO/LISO/LILO patterns
2. **TTFT vs input length** — scaling from 32 to 256 tokens
3. **Prefix caching impact** — TTFT with cache on vs off
4. **Decode time vs output length** — scaling from 32 to 768 output tokens
5. **Throughput vs concurrency** — output tok/s across concurrency levels 4-64

Deliverables:
- Clean plots for each experiment
- Short writeup interpreting the results (what the numbers mean, not just what they are)
- README update with architecture diagram, plots, and findings summary

This is the fastest path to a shareable, interview-ready artifact.

---

## Pre-reading: vLLM scheduler internals

Before starting Phase 2, read `vllm/core/scheduler.py`. One afternoon, no code to write. This is where vLLM decides which requests to prefill, which to decode, when to preempt (swap to CPU or recompute), and how to pack requests into a batch given the KV cache budget. Understanding *why* it makes each decision gives you the mental model for everything that follows — TRT-LLM's executor config will make more sense, and speculative decoding requires building a scheduler yourself.

---

## Phase 2 — TensorRT-LLM inference engine

Build a second inference backend using TensorRT-LLM, keeping the gateway, sidecar, and benchmarking framework unchanged. This proves the architecture is backend-agnostic and demonstrates deeper inference knowledge.

### 2a. Model compiler/builder
- Script to convert HuggingFace weights into a TensorRT engine binary
- Configuration surface: quantization (FP16, FP8, INT8), tensor parallelism, max batch size, max sequence lengths, CUDA graphs, paged KV cache

### 2b. Engine executor wrapper
- Load compiled TensorRT engine
- Submit requests via TRT-LLM's executor API
- Manage request lifecycles (streaming tokens, cancellation)
- Expose the same OpenAI-compatible interface the gateway already expects

### 2c. KV cache integration
- Configure and manage KV cache pools explicitly (pool sizes, block allocation, paged cache toggle)
- Wire into existing sidecar gRPC interface for KV offload

### 2d. Comparative benchmarks
- Run the same 5 experiments against TRT-LLM backend
- Produce side-by-side plots (vLLM vs TRT-LLM)
- Write up analysis explaining *why* the numbers differ (compile-time optimizations, CUDA graphs, memory management differences)

This is the highest-value phase for inference roles. "Here's how TTFT scales under vLLM vs TRT-LLM with the same model and input distribution" is a strong interview talking point.

---

## Phase 3 (optional) — Build core inference from scratch

Implement a minimal inference engine without vLLM or TRT-LLM to understand the internals:

- KV cache with paged attention (PagedAttention from scratch)
- Continuous batching scheduler
- Basic CUDA kernel integration for attention

High learning value. High time cost. Only pursue this if Phase 2 is done and you want to go deeper. Not required for most inference roles, but differentiating for the most technical ones.

---

## Phase 5 (if time allows) — Production hardening & scaling

Everything below is real engineering work but doesn't directly strengthen an inference-focused portfolio. Do it if you want to target ML platform/infra roles or if you've completed the phases above.

### Robustness (from plan_robustness.md)
- **Unified config system** — single `InferenceServerConfig` with cross-component validation (R1.1)
- **Structured error handling** — error codes, consistent error responses across services (R1.2)
- **Structured JSON logging** — replace `logging.basicConfig` with JSON formatter + contextvars for request IDs (R1.3)
- **Circuit breakers + retry with backoff** — gateway→engine and engine→sidecar resilience (R2.1)
- **Health check cascading** — gateway `/ready` checks engine health, not just its own process (R2.2)
- **Rate limiting at gateway** — token-bucket rate limiter (R2.4)
- **Registry file locking + atomic writes** — prevent corruption on concurrent writes (R3.1)
- **Input validation at gateway** — replace raw `dict` with proper Pydantic model (R3.5)
- **Request ID propagation** — X-Request-ID from gateway through engine and sidecar (R1.2)
- **Pre-flight checks** — validate GPU, disk space, config on startup (R5.1)
- **K8s probe separation** — `/healthz`, `/readyz`, `/startupz` endpoints (R5.2)
- **Gateway Prometheus metrics** — `/metrics` endpoint with request counts, latency histograms (R4.2)

### Kubernetes & scaling (from plan_scaling.md)
- Complete K8s manifests with proper probes, resource limits, secrets
- gRPC migration for gateway→engine path (HTTP/2 multiplexing, streaming, deadline propagation)
- Replace hardcoded `MODEL_SERVICE_MAP` with K8s service discovery
- HPA for gateway autoscaling
- Multi-pod routing with cache-aware affinity

### Observability
- OpenTelemetry distributed tracing across all three services (R4.1)
- Alerting rules for Prometheus (R4.2)

Full details in `docs/plan_robustness.md` and `docs/plan_scaling.md`.

---

## Follow-up projects

Standalone projects to pursue after this inference server is complete. Each one deepens a different dimension of inference engineering.

### Speculative decoding engine

Use a small draft model to propose N tokens, then verify all N in a single forward pass of the target model. Accepted tokens are "free" — trading one slow autoregressive step for one fast draft + one parallel verify.

- Run both models on a single GPU (e.g. Qwen2-0.5B as draft, Qwen2-1.5B as target — fits in 8GB VRAM)
- Implement the acceptance/rejection sampling algorithm
- Build a scheduler that coordinates draft and verify steps within the continuous batching loop
- Handle edge cases: what happens when the draft model is terrible for certain prompts
- Benchmark: measure acceptance rates, speedup vs draft model size, breakdown across prompt types

Doable locally. Strong benchmarking story ("speculative decoding gives 1.8x speedup on code generation but only 1.1x on creative writing because acceptance rate drops"). Systems-level Python work, no CUDA required.

### Disaggregated inference (prefill/decode separation)

Separate prefill (compute-bound) and decode (memory-bound) onto different hardware. Prefill nodes crunch through the prompt, serialize the KV cache, ship it to decode nodes optimized for memory bandwidth.

- KV cache serialization and transfer across nodes
- Scheduler that routes between prefill and decode pools
- Manage the latency budget for the network hop
- This is what Mooncake (Databricks), DistServe, and DeepSeek are doing right now

Requires multi-node access (e.g. two RunPod instances — one compute-optimized, one memory-optimized). Hard to do meaningfully on a single GPU since the whole point is exploiting different hardware profiles. The existing sidecar + gRPC KV offload architecture is a starting point.

### Custom inference engine with hand-written kernels

Build the forward pass from scratch: load safetensors weights manually, implement paged KV cache with a custom block allocator, write attention kernels in Triton or raw CUDA, build a continuous batching scheduler. Make it actually serve requests.

It won't compete with vLLM on performance. The point is understanding every single byte moving through the system. Requires learning CUDA/Triton. Hardest and longest of the three, but the most differentiating for deeply technical inference roles.

---

## What NOT to do

- **Deploy to RunPod** — nobody will use it, "deployed but unused" invites awkward interview questions
- **K8s multi-pod routing** — signals platform engineering, not inference engineering. The existing CRDs and control plane skeleton are enough to show awareness
- **Scope creep on the server** — the server is done enough. It's proof you can build, not the portfolio piece itself

---

## Interview value summary

| Phase | What it proves |
|-------|---------------|
| Complete single-GPU server | You can build end-to-end, not just scaffold |
| Benchmarking | You can measure, analyze, and reason about inference performance |
| TRT-LLM engine | You understand inference at the GPU level, not just API wrappers |
| Comparative analysis | You can evaluate tradeoffs across frameworks with real data |
| (Optional) From scratch | You understand the fundamental algorithms, not just the tools |
| (If time) Hardening & scaling | You can production-harden and scale serving infrastructure |
| **Follow-up: Speculative decoding** | You can optimize inference latency at the algorithm level |
| **Follow-up: Disaggregated inference** | You understand cutting-edge serving architectures |
| **Follow-up: Custom engine + kernels** | You understand every byte moving through the GPU |
