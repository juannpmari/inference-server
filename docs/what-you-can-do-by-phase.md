# What You Can Do — By Phase

A practical guide to system capabilities after each phase of development. **For the impatient: skip to the phase you care about.**

---

## Phase A: Make It Importable
**Status**: Prerequisite. Nothing user-facing yet.

### What You Get
- The codebase stops throwing `ModuleNotFoundError`
- You can `import` modules without errors
- Basic types and protocol definitions exist

### What You Can Do
- ✅ Run `python -c "from data_plane.inference.engine import engine"` without crashing
- ❌ Start the server
- ❌ Make requests
- ❌ Run any tests

### What's Missing
Everything. This is setup work.

---

## Phase B: Make It Configurable & Testable
**Status**: Still prerequisite. Nothing user-facing yet.

### What You Get
- Config modules with environment variable support
- Pytest infrastructure + basic test fixtures
- Makefile for common tasks

### What You Can Do
- ✅ `make install` — install all dependencies
- ✅ `make lint` — check code style
- ✅ `make test` — run unit tests (they'll mostly pass)
- ✅ Set env vars to override defaults: `ENGINE_MODEL_NAME="meta-llama/Llama-2-7b" python app.py`
- ❌ Start the server and have it work
- ❌ Make actual requests
- ❌ Do anything with models or inference

### What's Missing
The engine and sidecar are still broken (missing imports, incomplete code).

---

## Phase C: Fix the Engine ⭐ **FIRST USER-FACING MILESTONE**
**Status**: Core inference path works locally.

### What You Get
- ✅ **Working vLLM inference engine**
- ✅ **HTTP API** on `http://localhost:8080`
- ✅ **Model loading** from HuggingFace Hub
- ✅ **Health/readiness probes** for orchestration
- ✅ **Prometheus metrics** endpoint

### What You Can Do (Real Stuff!)

**1. Start the engine:**
```bash
ENGINE_MODEL_NAME="Qwen/Qwen2-0.5B" uvicorn data_plane.inference.engine.api:app --port 8080
```

**2. Check if it's ready:**
```bash
curl http://localhost:8080/ready
# Returns 200 if model is loaded, 503 if still loading
```

**3. Generate text (the main feature!):**
```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Returns:
# {"text": "The future of AI is very promising..."}
```

**4. Get metrics:**
```bash
curl http://localhost:8080/metrics
# Prometheus format counters for:
# - Requests total
# - Request latency
# - Time-to-first-token
# - Tokens generated
```

**5. Change the model (restart required):**
```bash
ENGINE_MODEL_NAME="meta-llama/Llama-2-7b" uvicorn ...
# Or: deploy a new pod with different config
```

### Limitations
- ⚠️ **Single model per instance** — change it via config/restart
- ⚠️ **No adapters (LoRA) yet** — only base models
- ⚠️ **No streaming** — `/inference` returns complete text at once
- ⚠️ **No concurrency control** — requests queue in FastAPI, vLLM batches them sequentially (no preemption)
- ⚠️ **No caching** — same prompt twice = duplicate computation
- ⚠️ **Single engine only** — no load balancing, no failover
- ⚠️ **Basic error handling** — no graceful degradation, no request timeout

### What's Next
You can now manually test inference, verify metrics, and prototype requests. If you scale requests, you'll queue up on the engine's async queue and eventually see timeouts. This is fine — that's what Phase K (robustness) fixes.

---

## Phase D: Fix the Sidecar (parallel with C)
**Status**: Model management works. Multi-model support ready.

### What You Get
- ✅ **Model loader service** (the sidecar)
- ✅ **Runtime model load/unload** without restarting engine
- ✅ **Model registry** — list what's loaded
- ✅ **Adapter support foundation** (LoRA ready)

### What You Can Do

**1. Load a model without restarting engine:**
```bash
curl -X POST http://localhost:8001/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2-1B"}'

# Engine can now use Qwen2-1B, even if it was using 0.5B before
```

**2. Unload a model (free VRAM):**
```bash
curl -X POST http://localhost:8001/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2-0.5B"}'
```

**3. See what's loaded:**
```bash
curl http://localhost:8001/registry/models
# Returns: [{"model_id": "Qwen/Qwen2-1B", "status": "loaded", ...}]
```

**4. Check sidecar health:**
```bash
curl http://localhost:8001/health
```

### Limitations
- ⚠️ **Still single engine** — you can hot-swap models, but one engine handles all requests
- ⚠️ **No multi-model support in engine yet** — engine can only use one model at a time (need to load/unload to switch)
- ⚠️ **No adapter (LoRA) functionality yet** — just the plumbing
- ⚠️ **Models stored locally only** — no distributed cache
- ⚠️ **Limited registry** — just in-memory, doesn't persist across restarts

### How This Combines with Phase C
Now you have **two services**:
- **Engine** (port 8080): inference
- **Sidecar** (port 8001): model management

You can still do everything from Phase C, **plus** change models on-the-fly:
1. Start engine with model A
2. Call sidecar to load model B
3. Call engine to infer with model B
4. No engine restart

---

## Phase E: Docker Infrastructure
**Status**: Everything runs in containers locally. Integration ready.

### What You Get
- ✅ **Docker Compose setup** — one `docker-compose up` runs everything
- ✅ **All services containerized** (gateway, engine, sidecar, Redis, Prometheus)
- ✅ **Shared volumes** for models
- ✅ **Network connectivity** between services

### What You Can Do

**Start the whole system:**
```bash
docker-compose up -d
```

**Now all services are running:**
```
gateway     (http://localhost:8000) — main API
engine      (http://localhost:8080) — inference engine
sidecar     (http://localhost:8001) — model management
redis       (6379)                  — L2 cache backend
prometheus  (http://localhost:9090) — metrics dashboard
```

**Check Prometheus dashboard:**
```
http://localhost:9090/graph
# See all metrics from all services
```

### Limitations
- ⚠️ **Gateway doesn't route yet** — it's there, but doesn't actually balance requests to multiple engines
- ⚠️ **Redis is empty** — L2 cache plumbing exists but isn't used yet
- ⚠️ **Metrics only from engine** — sidecar metrics may not be complete
- ⚠️ **One engine pod only** — you can scale later, but routing isn't smart yet

### How This Changes Your Workflow
Instead of:
```bash
uvicorn data_plane.inference.engine.api:app --port 8080
# (in another terminal)
uvicorn data_plane.inference.sidecar.api:app --port 8001
```

Now you do:
```bash
docker-compose up -d
# Everything auto-starts, logs stream to terminal, Ctrl+C stops all
```

---

## Phase F: Streaming
**Status**: Tokens arrive incrementally instead of all at once.

### What You Get
- ✅ **Server-Sent Events (SSE) endpoint**
- ✅ **Streaming text generation**
- ✅ **Client disconnect handling** (cancels generation)

### What You Can Do

**Stream tokens as they arrive (instead of waiting for full response):**
```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story",
    "max_tokens": 100,
    "stream": true
  }'

# Response comes in chunks:
# data: The
# data:  quick
# data:  brown
# data:  fox
# ...
```

**In a browser or JS client:**
```javascript
const eventSource = new EventSource('/inference?stream=true&prompt=...');
eventSource.onmessage = (e) => {
  console.log('Token:', e.data);
};
```

**Abort mid-generation (client disconnects):**
- Close your curl / browser tab
- Engine cancels the request and stops generating
- VRAM freed immediately

### Limitations
- ⚠️ **No request timeout yet** — if a request hangs, it hangs forever
- ⚠️ **No queue limits** — you can't reject the 100th concurrent request

### How This Changes UX
Instead of:
```
[waiting 5 seconds...]
Full response: "The future of AI is very promising because..."
```

Now:
```
[immediate] The
[+50ms] future
[+50ms] of
[+50ms] AI
[+50ms] is
[...]
```

Users see partial results **immediately**. Much better for UI/chatbot apps.

---

## Phase G: LoRA (Low-Rank Adapters)
**Status**: Multiple models via parameter-efficient fine-tuning. Shared base weights.

### What You Get
- ✅ **Load LoRA adapters** without duplicating base model
- ✅ **Multiple specialized models** from one base
- ✅ **Switch adapters on-the-fly**

### What You Can Do

**Example: Medical + Legal domain specialists**

**1. Load base model:**
```bash
curl -X POST http://localhost:8001/load \
  -d '{"model_id": "meta-llama/Llama-2-7b"}'
```

**2. Load adapters (no extra VRAM for base model):**
```bash
curl -X POST http://localhost:8001/load-adapter \
  -d '{"adapter_id": "medical-llama-lora", "base_model": "meta-llama/Llama-2-7b"}'

curl -X POST http://localhost:8001/load-adapter \
  -d '{"adapter_id": "legal-llama-lora", "base_model": "meta-llama/Llama-2-7b"}'
```

**3. Use adapters for inference:**
```bash
curl -X POST http://localhost:8080/inference \
  -d '{
    "prompt": "The patient presented with...",
    "adapter_id": "medical-llama-lora",
    "max_tokens": 200
  }'

curl -X POST http://localhost:8080/inference \
  -d '{
    "prompt": "The contract states that...",
    "adapter_id": "legal-llama-lora",
    "max_tokens": 200
  }'
```

### Limitations
- ⚠️ **Max 10 adapters** per base model (hardcoded, can change)
- ⚠️ **One adapter active at a time** (per request)
- ⚠️ **Adapters must be from HuggingFace** (no local fine-tuning yet)
- ⚠️ **No adapter caching** — loaded into VRAM always

### How This Enables New Use Cases
Instead of:
```
One big model for everything
→ Generic responses (not great for specialized domains)
```

Now:
```
One base model (7B parameters)
+ 3 specialized adapters (3% each = 210M total)
→ Specialized responses, domain-specific behavior, shared base weights
```

Cost-effective multi-task inference.

---

## Phase H: Registry Persistence
**Status**: Models and adapters survive restart.

### What You Get
- ✅ **Registry stored to disk** (JSON file)
- ✅ **Models/adapters auto-load on restart**
- ✅ **Metadata** (tags, preferred device, warmup status)

### What You Can Do

**Load models, then restart:**
```bash
# Start system
docker-compose up -d

# Load some models
curl -X POST http://localhost:8001/load \
  -d '{"model_id": "Qwen/Qwen2-1B", "tags": ["production", "fast"]}'

# Restart
docker-compose down && docker-compose up -d

# Models are back! Ready to infer immediately.
```

**Query registry with metadata:**
```bash
curl http://localhost:8001/registry/models
# Returns:
# [
#   {
#     "model_id": "Qwen/Qwen2-1B",
#     "local_path": "/models/qwen2-1b",
#     "status": "loaded",
#     "tags": ["production", "fast"],
#     "loaded_at": "2025-02-17T10:30:45Z"
#   }
# ]
```

### Limitations
- ⚠️ **Registry persists, but artifacts don't auto-fetch** — models are stored locally, not re-downloaded on restart
- ⚠️ **Metadata only** — tags are strings, no complex queries
- ⚠️ **Single file backend** — not suitable for 1000s of models

### How This Affects Operations
Instead of:
```
Restart sidecar
→ All models unloaded
→ Need to manually load them again
→ Slow warmup time
```

Now:
```
Restart sidecar
→ Models auto-load from disk
→ Services are ready-to-serve immediately
→ No manual intervention
```

Great for deployments and CI/CD.

---

## Phase I: KV Cache L1 (CPU Memory Cache)
**Status**: First-level cache reduces redundant computation.

### What You Get
- ✅ **L1 cache** (CPU DRAM) for KV pairs
- ✅ **LRU eviction** when full
- ✅ **Cache hit/miss metrics**

### What You Can Do

**Automatic cache reuse (transparent to user):**
```bash
# First request
curl -X POST http://localhost:8080/inference \
  -d '{
    "prompt": "The quick brown fox jumps over",
    "max_tokens": 10
  }'
# → Computes full KV cache (slow)

# Second request with same prefix
curl -X POST http://localhost:8080/inference \
  -d '{
    "prompt": "The quick brown fox jumps over the lazy dog and",
    "max_tokens": 10
  }'
# → Reuses cached KVs from prefix (faster!)
```

**Check cache stats:**
```bash
curl http://localhost:8080/metrics | grep cache_l1
# cache_l1_hits_total 5
# cache_l1_misses_total 2
# cache_l1_capacity_bytes 536870912  (512 MB)
# cache_l1_used_bytes 250000000
```

### Limitations
- ⚠️ **Only within one engine pod** — doesn't help if requests go to different pods
- ⚠️ **Same model only** — switching models clears cache
- ⚠️ **Prefix-based** — cache hit only if you reuse the exact prompt prefix
- ⚠️ **No persistence** — cache is in-memory, lost on restart
- ⚠️ **L2 (Redis) not connected yet** — cached entries don't spill to Redis

### How This Improves Performance
Same prefix 10 times:
```
Without cache:  10 × 3 seconds = 30 seconds
With L1 cache:  1 × 3 seconds + 9 × 0.5 seconds = 7.5 seconds (4× faster!)
```

Good for chat apps where users ask follow-ups.

---

## Phase J: Cache-Aware Routing
**Status**: Multiple engines with intelligent request distribution.

### What You Get
- ✅ **Gateway routes to multiple engines**
- ✅ **Cache-hit preference** (routes to engine with cached prefix)
- ✅ **Fallback to load-aware routing** (least connections)

### What You Can Do

**Scale to multiple engines (3 pods):**
```bash
# Register engines (or auto-discover in K8s)
curl -X POST http://localhost:8000/register \
  -d '{"id": "engine-1", "url": "http://engine-1:8080"}'

curl -X POST http://localhost:8000/register \
  -d '{"id": "engine-2", "url": "http://engine-2:8080"}'

curl -X POST http://localhost:8000/register \
  -d '{"id": "engine-3", "url": "http://engine-3:8080"}'
```

**Make requests to gateway (not direct engine):**
```bash
# Instead of: curl http://engine-1:8080/inference
# Do this:
curl -X POST http://localhost:8000/inference \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# Gateway routes based on:
# 1. Which engine has cached prefix? → send there (cache hit!)
# 2. Otherwise, which has fewest pending requests? → send there
# 3. Return result as if it came from a single engine
```

### Limitations
- ⚠️ **Cache only shared across 1 instance per prefix** — you can't merge caches from 2 engines
- ⚠️ **Engines still isolated** — no shared KV store yet
- ⚠️ **Prefix hash collision possible** (unlikely, but ~1 in 2^64)
- ⚠️ **No cross-pod L2 cache yet** — Redis isn't shared for KV cache

### How This Changes Scalability
Instead of:
```
Bottleneck: Single engine can only batch so fast
Solution: Hope request patterns are sequential (not great)
```

Now:
```
Solution: Route to best engine based on cache state
Result: Better utilization of cluster, fewer cache misses
```

For a 10-request burst on 3 engines:
```
Naive round-robin: Each engine gets ~3.3 requests, only 1-2 benefit from cache
Smart routing: 7 requests hit existing cache, 3 miss (better!)
```

---

## Phase K: Robustness
**Status**: System can handle edge cases gracefully.

### What You Get
- ✅ **Queue limits** (HTTP 429 when full)
- ✅ **Request timeouts** (HTTP 504 if taking too long)
- ✅ **Graceful shutdown** (drain in-flight requests, no sudden drops)

### What You Can Do

**Overload the engine safely:**
```bash
# Send 15 concurrent requests, engine max queue is 10:
for i in {1..15}; do
  curl -X POST http://localhost:8080/inference \
    -d '{"prompt": "Tell a story", "max_tokens": 1000}' &
done

# First 10 start processing
# 11-15 get:
# HTTP 429 Too Many Requests
# Retry-After: 5
# {"error": "Queue full, try again in 5 seconds"}
```

**Timeout protection:**
```bash
# If a request would take >30 seconds, kill it
# (config: REQUEST_TIMEOUT_SECONDS=30)

curl -X POST http://localhost:8080/inference \
  -d '{
    "prompt": "...",
    "max_tokens": 100000  # would take forever
  }'

# After 30 seconds:
# HTTP 504 Gateway Timeout
# {"error": "Request exceeded 30s timeout"}
```

**Graceful shutdown:**
```bash
# In-flight requests complete (up to shutdown grace period)
docker-compose down  # Sends SIGTERM to all containers

# Engine behavior:
# 1. Stop accepting new requests (501)
# 2. Wait up to 30 seconds for in-flight to finish
# 3. Terminate
# (No requests dropped if they complete in time)
```

### Limitations
- ⚠️ **Still single-pod** — queue limits are per-engine, not global
- ⚠️ **No auto-retry on timeout** — client must retry manually
- ⚠️ **No circuit breaker** — if engine is broken, gateway retries indefinitely

### How This Protects Your System
Without robustness:
```
User sends 1000 requests
→ Engine queues all 1000
→ Memory bloat
→ OOMKilled
→ Whole service down
```

With robustness:
```
User sends 1000 requests
→ Engine queues 10, rejects 990 with 429
→ Client retries in a few seconds
→ Controlled load, system stable
```

---

## Phase L: Sidecar Completion
**Status**: Model management is production-ready.

### What You Get
- ✅ **Download retries** (3× exponential backoff)
- ✅ **Model warmup** (engine pre-populates cache after load)
- ✅ **gRPC support** (sidecar talks to engine for KV cache coordination)
- ✅ **Structured JSON logging** (easy to parse logs)

### What You Can Do

**Reliable model downloads:**
```bash
curl -X POST http://localhost:8001/load \
  -d '{"model_id": "meta-llama/Llama-2-13b"}'

# If network hiccup:
# - Retry 1: after 1 second
# - Retry 2: after 2 seconds
# - Retry 3: after 4 seconds
# - If all fail: return error

# Atomic download (temp file → rename on success)
```

**Auto-warm cache after load:**
```bash
curl -X POST http://localhost:8001/load \
  -d '{"model_id": "medical-model", "warmup": true}'

# Sidecar automatically sends 5 sample prompts to engine:
# - "The patient presented with..."
# - "Diagnosis: ..."
# (etc.)
# → First real request benefits from pre-populated cache
```

**Monitor via JSON logs:**
```
tail -f sidecar.log | jq .
# {
#   "timestamp": "2025-02-17T10:30:45Z",
#   "level": "INFO",
#   "message": "Model loaded",
#   "model_id": "qwen2-1b",
#   "duration_seconds": 2.5,
#   "status": "success"
# }
```

### Limitations
- ⚠️ **Warmup is hardcoded** — same prompts for all models
- ⚠️ **gRPC only for logging** — not for actual KV cache sharing yet
- ⚠️ **Download retries work, but slow** — 7 seconds worst case for 3 retries

### How This Improves Reliability
Instead of:
```
Network blip during model load
→ Sidecar crashes
→ Manual restart + reload needed
→ Downtime
```

Now:
```
Network blip during model load
→ Automatic retry in <1 second
→ Continues working
→ Metrics logged, no human intervention
```

---

## Phase M: Batching Metric
**Status**: Final polish. Metrics complete.

### What You Get
- ✅ **Batch size histogram** (how many requests per step)
- ✅ **Step latency tracking** (vLLM engine.step() timing)

### What You Can Do

**Monitor batch efficiency:**
```bash
curl http://localhost:8080/metrics | grep batch_size
# engine_batch_size_bucket{le="1.0"} 100
# engine_batch_size_bucket{le="2.0"} 95
# engine_batch_size_bucket{le="4.0"} 85
# engine_batch_size_bucket{le="inf"} 120

# Interpretation:
# - 100 steps with 1 request (low utilization)
# - 95 steps with 2 requests
# - 85 steps with 4 requests (best case)
# - Never batched >4 (not much concurrency)
```

**In a Prometheus dashboard:**
```
engine_batch_size_bucket (histogram)
→ Shows if you're under-utilizing GPU (batch_size=1 often)
→ Tells you when to add more concurrent clients
```

### How This Helps Operations
Instead of:
```
"Why is the GPU at 40% utilization?"
→ No visibility
→ Guessing game
```

Now:
```
"GPU at 40% utilization because batch_size=1 (no concurrency)"
→ Add more clients / concurrent requests
→ Watch batch_size improve, GPU utilization climb
```

---

## Summary: Feature Timeline

| Phase | Core Capability | User-Facing Change |
|-------|-----------------|-------------------|
| A-B   | Setup           | (none, prerequisite) |
| **C** | **Inference**   | ✅ Generate text from prompts |
| **D** | **Model Management** | ✅ Hot-swap models without restart |
| **E** | **Containers**  | ✅ `docker-compose up` for full stack |
| **F** | **Streaming**   | ✅ Tokens appear incrementally |
| **G** | **LoRA**        | ✅ Multiple domain-specific adapters |
| **H** | **Persistence** | ✅ Models survive restart |
| **I** | **L1 Cache**    | ✅ Repeated prompts are ~4× faster |
| **J** | **Routing**     | ✅ Scale to 3+ engines with cache awareness |
| **K** | **Robustness**  | ✅ Graceful overload handling |
| **L** | **Reliability** | ✅ Retries, warmup, structured logs |
| **M** | **Monitoring**  | ✅ Full metrics for batch optimization |

---

## Your Next Move

**Want to start?** → Phase C is your first win. You'll have a working inference engine.

**Want multi-model?** → Phase C + D. Load/unload models on-the-fly.

**Want to scale?** → Phase C + D + E + J. Multiple engines with smart routing.

**Want production-grade?** → Complete Phase A–M. Full-featured distributed inference system.
