import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CollectorRegistry, REGISTRY
import time

from data_plane.inference.engine.config import EngineConfig
from data_plane.inference.engine import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=False, description="Stream tokens as they're generated")
    adapter_identifier: Optional[str] = Field(default=None, description="LoRA adapter ID")
    adapter_version: Optional[str] = Field(default=None, description="LoRA adapter version")


class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    duration_seconds: float


# Global state
_engine = None
_batching_loop = None
_config = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting engine...")
    # Startup
    global _engine, _batching_loop, _config
    try:
        _config = EngineConfig()

        if _config.enable_engine_mock:
            logger.info("Using MOCK engine (no GPU)")
            from data_plane.inference.engine.mock_engine import MockLLMEngine
            _engine = MockLLMEngine(_config)
        else:
            logger.info("Using REAL vLLM engine")
            model_path = await _wait_for_sidecar_model(_config)
            from data_plane.inference.engine.engine import Engine
            _engine = Engine(_config, model_path=model_path)

        _batching_loop = asyncio.create_task(_engine.continuous_batching_loop())
        logger.info("Engine startup complete")
    except Exception as e:
        logger.error(f"Engine startup failed: {e}")
        raise

    yield

    # Shutdown
    try:
        logger.info("Shutting down engine...")
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


@app.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(request: InferenceRequest):
    """
    Generate text from a prompt.
    Validates input and handles errors gracefully.
    """
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

    if len(_engine.request_futures) >= _config.max_pending:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Queue full, too many pending requests",
            headers={"Retry-After": "1"}
        )

    try:
        start_time = time.time()
        model_id = _config.model_name

        # Add request to engine
        result = await asyncio.wait_for(
            _engine.add_request(
                prompt=request.prompt,
                adapter_identifier=request.adapter_identifier,
                adapter_version=request.adapter_version,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ),
            timeout=300.0  # 5 minute timeout
        )

        duration = time.time() - start_time

        # Record metrics
        metrics.engine_requests_total.labels(model=model_id, status="success").inc()
        metrics.engine_request_duration_seconds.labels(model=model_id).observe(duration)

        # Parse result to count tokens
        tokens_generated = len(result.split()) if isinstance(result, str) else 0

        metrics.engine_tokens_generated_total.labels(model=model_id).inc(tokens_generated)

        return InferenceResponse(
            text=result,
            tokens_generated=tokens_generated,
            duration_seconds=duration
        )

    except asyncio.TimeoutError:
        metrics.engine_requests_total.labels(model=_config.model_name, status="timeout").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request generation timed out"
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        metrics.engine_requests_total.labels(model=_config.model_name, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8")
