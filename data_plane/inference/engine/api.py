import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CollectorRegistry, REGISTRY
import time

from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine import metrics
from shared.monitoring import SessionCollector, TimingInfo, RequestRecord
from shared.monitoring.gpu import GPUMonitor
from shared.monitoring.storage import LocalJSONLStore, BackgroundFlusher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    stream: bool = Field(default=False, description="Stream tokens as they're generated")
    adapter_identifier: Optional[str] = Field(default=None, description="LoRA adapter ID")
    adapter_version: Optional[str] = Field(default=None, description="LoRA adapter version")


class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    duration_seconds: float
    prompt_tokens: int = 0
    finish_reason: str = "stop"


# Global state
_engine = None
_batching_loop = None
_init_task = None
_config = None
_collector: Optional[SessionCollector] = None
_gpu_monitor: Optional[GPUMonitor] = None
_flusher: Optional[BackgroundFlusher] = None

# Track high-water mark of vLLM's internal num_requests_waiting gauge.
_vllm_waiting_lock = threading.Lock()
_vllm_max_requests_waiting: float = 0.0

_VLLM_WAITING_NAMES = {"vllm:num_requests_waiting", "vllm_num_requests_waiting"}


def _sample_vllm_waiting() -> None:
    """Read vllm:num_requests_waiting from the Prometheus registry and update the max."""
    global _vllm_max_requests_waiting
    try:
        for metric in REGISTRY.collect():
            if metric.name in _VLLM_WAITING_NAMES:
                for sample in metric.samples:
                    with _vllm_waiting_lock:
                        if sample.value > _vllm_max_requests_waiting:
                            _vllm_max_requests_waiting = sample.value
                return
    except Exception:
        pass


async def _wait_for_sidecar_model(config: EngineConfig) -> str:
    """Poll the sidecar registry until the model is loaded, then return its local_path."""
    url = f"{config.sidecar_url}/registry/models"
    deadline = time.monotonic() + config.sidecar_timeout
    logger.info(f"Waiting for sidecar to finish loading model '{config.model_name}'...")

    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(url, timeout=5.0)
                resp.raise_for_status()
                registry = resp.json()
                entry = registry.get(config.model_name)
                if entry and entry.get("status") == "loaded":
                    local_path = entry["local_path"]
                    logger.info(f"Resolved model to local path: {local_path}")
                    return local_path
                if entry:
                    logger.info(f"Model found but status='{entry.get('status')}', waiting...")
                else:
                    logger.info("Model not yet in sidecar registry, waiting...")
            except httpx.RequestError as e:
                logger.info(f"Sidecar not reachable yet ({e}), waiting...")

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Sidecar did not load model '{config.model_name}' within {config.sidecar_timeout}s"
                )
            await asyncio.sleep(config.sidecar_poll_interval)


async def _init_engine(config: EngineConfig):
    """Background task: wait for sidecar, create engine, start batching loop."""
    global _engine, _batching_loop
    try:
        if config.enable_engine_mock:
            logger.info("Using MOCK engine (no GPU)")
            from data_plane.inference.engine.mock_engine import MockLLMEngine
            _engine = MockLLMEngine(config, collector=_collector)
        else:
            logger.info("Using REAL vLLM engine")
            model_path = await _wait_for_sidecar_model(config)
            from data_plane.inference.engine.engine import Engine
            _engine = await asyncio.to_thread(Engine, config, model_path=model_path, collector=_collector)

        logger.info(f"_engine set to: {type(_engine)}, id={id(_engine)}")
        _batching_loop = asyncio.create_task(_engine.continuous_batching_loop())
        logger.info("Engine startup complete")
    except Exception as e:
        logger.error(f"Engine startup failed: {e}", exc_info=True)


def _gpu_on_sample(snapshot: dict):
    """Callback from GPUMonitor to update Prometheus gauges."""
    metrics.engine_gpu_compute_utilization_percent.labels(device="0").set(snapshot["compute_utilization_pct"])
    metrics.engine_gpu_memory_used_bytes.labels(device="0").set(snapshot["memory_used_bytes"])
    metrics.engine_gpu_memory_total_bytes.labels(device="0").set(snapshot["memory_total_bytes"])
    metrics.engine_gpu_power_watts.labels(device="0").set(snapshot["power_watts"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting engine...")
    global _init_task, _config, _collector, _gpu_monitor, _flusher
    _config = EngineConfig()

    # Initialize monitoring
    def _lora_state():
        if _engine and hasattr(_engine, 'lora_manager') and _engine.lora_manager:
            return {
                "loaded_count": _engine.lora_manager.loaded_count,
                "loaded_keys": _engine.lora_manager.loaded_keys,
            }
        return {"loaded_count": 0, "loaded_keys": []}

    _collector = SessionCollector(
        maxlen=_config.monitoring_buffer_size,
        lora_state_fn=_lora_state,
    )

    # GPU monitoring
    if _config.gpu_monitor_enabled:
        _gpu_monitor = GPUMonitor(
            poll_interval=_config.gpu_poll_interval,
            device_index=_config.gpu_device_index,
            on_sample=_gpu_on_sample,
        )
        _gpu_monitor.start()
        logger.info(f"GPUMonitor started (available={_gpu_monitor.available})")

    # Storage persistence
    store = LocalJSONLStore(_config.monitoring_local_store_path)
    _flusher = BackgroundFlusher(
        collector=_collector,
        backend=store,
        interval=_config.monitoring_flush_interval,
    )
    _flusher.start()

    # Launch engine init in background so the server starts accepting requests immediately
    _init_task = asyncio.create_task(_init_engine(_config))

    yield

    # Shutdown
    try:
        logger.info("Shutting down engine...")
        if _flusher:
            await _flusher.stop()
        if _gpu_monitor:
            await _gpu_monitor.stop()
        if _init_task and not _init_task.done():
            _init_task.cancel()
            try:
                await _init_task
            except asyncio.CancelledError:
                pass
        if _batching_loop:
            _batching_loop.cancel()
            try:
                await _batching_loop
            except asyncio.CancelledError:
                pass
        logger.info("Engine shutdown complete")
    except Exception as e:
        logger.error(f"Engine shutdown error: {e}")


app = FastAPI(title="Engine", lifespan=lifespan)


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    if _engine is None:
        logger.info(f"Health check: _engine is None, _init_task done={_init_task.done() if _init_task else 'no task'}, _init_task exception={_init_task.exception() if _init_task and _init_task.done() else 'N/A'}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "initializing"}
        )
    return {"status": "healthy"}


@app.get("/ready", tags=["health"])
async def ready():
    """Readiness check endpoint"""
    if _engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "engine_not_initialized"}
        )

    if not _engine.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "model_loading"}
        )

    return {"status": "ready"}


def _check_engine_ready():
    """Raise HTTPException if engine is not ready."""
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    if not _engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    if len(_engine.request_futures) + len(getattr(_engine, "request_queues", {})) >= _config.max_pending:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Queue full, too many pending requests",
            headers={"Retry-After": "1"},
        )


@app.post("/generate", response_model=InferenceResponse, tags=["inference"])
async def generate(request: InferenceRequest):
    """Generate text from a prompt (internal endpoint called by gateway)."""
    _check_engine_ready()

    model_id = _config.model_name

    if _collector:
        _collector.inc_queue_depth()
    metrics.engine_pending_requests.labels(model=model_id).inc()
    _sample_vllm_waiting()

    try:
        start_time = time.time()

        result = await asyncio.wait_for(
            _engine.add_request(
                prompt=request.prompt,
                adapter_identifier=request.adapter_identifier,
                adapter_version=request.adapter_version,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                seed=request.seed,
            ),
            timeout=_config.sidecar_timeout,
        )

        duration = time.time() - start_time

        metrics.engine_requests_total.labels(model=model_id, status="success").inc()
        metrics.engine_request_duration_seconds.labels(model=model_id).observe(duration)

        tokens_generated = len(_engine.tokenize(result)) if isinstance(result, str) else 0
        input_tokens = len(_engine.tokenize(request.prompt))

        metrics.engine_tokens_generated_total.labels(model=model_id).inc(tokens_generated)

        if request.adapter_identifier:
            metrics.engine_lora_requests_total.labels(adapter=request.adapter_identifier).inc()

        return InferenceResponse(
            text=result,
            tokens_generated=tokens_generated,
            duration_seconds=duration,
            prompt_tokens=input_tokens,
            finish_reason="stop",
        )

    except asyncio.TimeoutError:
        metrics.engine_requests_total.labels(model=model_id, status="timeout").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request generation timed out",
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        metrics.engine_requests_total.labels(model=model_id, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
    finally:
        _sample_vllm_waiting()
        if _collector:
            _collector.dec_queue_depth()
        metrics.engine_pending_requests.labels(model=model_id).dec()


@app.post("/generate/stream", tags=["inference"])
async def generate_stream(request: InferenceRequest):
    """Stream tokens via SSE (internal endpoint called by gateway)."""
    _check_engine_ready()

    import json as _json

    async def _event_generator():
        queue = await _engine.add_streaming_request(
            prompt=request.prompt,
            adapter_identifier=request.adapter_identifier,
            adapter_version=request.adapter_version,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=request.seed,
        )
        while True:
            item = await queue.get()
            if item is None:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {_json.dumps(item)}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


class ChatTemplateRequest(BaseModel):
    messages: list[dict] = Field(..., description="List of chat messages")
    add_generation_prompt: bool = Field(default=True)


class ChatTemplateResponse(BaseModel):
    prompt: str


@app.post("/chat/apply_template", response_model=ChatTemplateResponse, tags=["inference"])
async def apply_template(request: ChatTemplateRequest):
    """Render messages through the model's chat template."""
    _check_engine_ready()
    prompt = _engine.apply_chat_template(
        request.messages, add_generation_prompt=request.add_generation_prompt
    )
    return ChatTemplateResponse(prompt=prompt)


@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8")


@app.get("/metrics_summary", tags=["monitoring"])
async def metrics_summary():
    """Aggregated metrics summary endpoint."""
    if _collector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring not initialized"
        )

    model_id = _config.model_name if _config else None
    summary = _collector.get_summary(model_id=model_id)

    # Merge GPU data
    if _gpu_monitor:
        gpu_summary = _gpu_monitor.get_summary()
        summary["gpu"] = gpu_summary
        # Compute tokens_per_watt
        output_tps = summary.get("throughput", {}).get("output_tokens_per_second", 0.0)
        power = gpu_summary.get("current", {}).get("power_watts", 0.0)
        summary["gpu"]["tokens_per_watt"] = round(output_tps / power, 4) if power > 0 else 0.0

    # Override queue stats with vLLM's engine-level view.
    _sample_vllm_waiting()
    vllm_waiting_now = None
    try:
        for metric in REGISTRY.collect():
            if metric.name in _VLLM_WAITING_NAMES:
                for sample in metric.samples:
                    vllm_waiting_now = sample.value
                break
    except Exception:
        pass
    with _vllm_waiting_lock:
        max_waiting = _vllm_max_requests_waiting
    summary["queue"]["current_depth"] = int(vllm_waiting_now) if vllm_waiting_now is not None else summary["queue"]["current_depth"]
    summary["queue"]["max_depth"] = int(max_waiting)

    # Read vLLM prefix-cache counter and our own input-token histogram
    # from the shared Prometheus registry to compute an approximate hit rate.
    # vLLM exposes "vllm:prefix_cache_hits_total" (cached token count);
    # prometheus_client normalises colons to underscores in metric.name.
    prefix_cache_hits = None
    total_input_tokens = None
    gpu_cache_usage = None
    _PREFIX_HITS = {"vllm:prefix_cache_hits_total", "vllm_prefix_cache_hits_total"}
    _GPU_CACHE = {"vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc"}
    try:
        for metric in REGISTRY.collect():
            name = metric.name
            if name in _PREFIX_HITS or f"{name}_total" in _PREFIX_HITS:
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        prefix_cache_hits = sample.value
            elif name in _GPU_CACHE:
                for sample in metric.samples:
                    gpu_cache_usage = sample.value
            elif name == "engine_input_tokens_per_request":
                for sample in metric.samples:
                    if sample.name.endswith("_sum"):
                        total_input_tokens = sample.value
    except Exception as e:
        logger.warning(f"Failed to read vLLM cache metrics: {e}")

    vllm_cache = {}
    if prefix_cache_hits is not None:
        vllm_cache["prefix_cache_hits_tokens"] = prefix_cache_hits
        if total_input_tokens and total_input_tokens > 0:
            vllm_cache["hit_rate"] = round(prefix_cache_hits / total_input_tokens, 4)
    if gpu_cache_usage is not None:
        vllm_cache["gpu_cache_usage_perc"] = gpu_cache_usage
    if vllm_cache:
        summary.setdefault("kv_cache", {}).update(vllm_cache)

    # Merge KV cache data from sidecar (graceful degradation)
    if _config and _config.sidecar_url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{_config.sidecar_url}/cache/stats")
                if resp.status_code == 200:
                    sidecar_stats = resp.json()
                    sidecar_stats.pop("hit_rate", None)  # sidecar hit_rate is broken; use vLLM's
                    summary.setdefault("kv_cache", {}).update(sidecar_stats)
        except Exception as e:
            logger.warning(f"Failed to fetch sidecar cache stats: {e}")

    return summary
