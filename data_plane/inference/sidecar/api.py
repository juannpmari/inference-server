import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import REGISTRY, generate_latest

from data_plane.inference.sidecar import metrics
from data_plane.inference.sidecar.artifact_manager import ArtifactManager
from data_plane.inference.sidecar.config import SidecarConfig
from data_plane.inference.sidecar.kv_block_registry import KVBlockRegistry

logger = logging.getLogger(__name__)

# Global state
_manager: Optional[ArtifactManager] = None
_config: Optional[SidecarConfig] = None
_kv_registry: Optional[KVBlockRegistry] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _config, _kv_registry
    logger.info("Starting sidecar...")

    _config = SidecarConfig()
    _manager = ArtifactManager(config=_config)
    _kv_registry = KVBlockRegistry()

    # Load initial model in background
    _manager.model_registry[_config.initial_model] = {
        "model_id": _config.initial_model,
        "version": _config.initial_model_version,
        "status": "downloading",
    }

    async def _initial_load():
        try:
            path = await _manager.load_model(
                model_identifier=_config.initial_model,
                version=_config.initial_model_version,
            )
            _manager.is_ready = True
            metrics.sidecar_resident_models.set(len(_manager.model_registry))
            logger.info(f"Initial model loaded at {path}")
        except Exception as e:
            logger.error(f"Initial model load failed: {e}")

    asyncio.create_task(_initial_load())
    logger.info("Sidecar startup complete")

    yield

    logger.info("Sidecar shutdown complete")


app = FastAPI(title="Sidecar", lifespan=lifespan)


@app.get("/health", tags=["health"])
async def health_check():
    """Liveness probe."""
    if _manager is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "initializing"},
        )
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Readiness probe — only ready when initial model is loaded."""
    if _manager is None or not _manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact Manager loading initial model.",
        )
    return {
        "status": "ready",
        "resident_models": list(_manager.model_registry.keys()),
    }


@app.post("/load/{model_identifier:path}", tags=["models"])
async def load_model_route(model_identifier: str, version: str = "latest"):
    """Accept a model load request and process it in the background."""
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    # Already loaded — return immediately
    existing = _manager.model_registry.get(model_identifier)
    if existing and existing.get("version") == version and existing.get("status") == "loaded":
        return {"status": "loaded", "model_identifier": model_identifier, "local_path": existing["local_path"]}

    # Already downloading — don't start a second task
    if existing and existing.get("status") == "downloading":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "downloading", "model_identifier": model_identifier},
        )

    # Mark as downloading and kick off background task
    _manager.model_registry[model_identifier] = {
        "model_id": model_identifier,
        "version": version,
        "status": "downloading",
    }

    async def _background_load():
        try:
            start = time.time()
            local_path = await _manager.load_model(model_identifier, version)
            duration = time.time() - start
            metrics.sidecar_model_load_duration_seconds.labels(model=model_identifier, source="huggingface").observe(
                duration
            )
            metrics.sidecar_resident_models.set(len(_manager.model_registry))
            logger.info(f"Background load complete: {model_identifier} at {local_path}")
        except Exception as e:
            logger.error(f"Background load failed for {model_identifier}: {e}")
            _manager.model_registry.pop(model_identifier, None)

    asyncio.create_task(_background_load())

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "downloading", "model_identifier": model_identifier},
    )


@app.post("/unload/{model_identifier:path}", tags=["models"])
async def unload_model_route(model_identifier: str):
    """Remove a model from the registry."""
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    if model_identifier not in _manager.model_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_identifier} not resident.",
        )

    _manager.unload_model(model_identifier)
    metrics.sidecar_resident_models.set(len(_manager.model_registry))
    return {"status": "success", "model_identifier": model_identifier}


@app.get("/registry/models", tags=["registry"])
async def get_models():
    """Returns the current resident models."""
    if _manager is None:
        return []
    return _manager.model_registry


@app.post("/adapter/unload/{adapter_identifier:path}", tags=["adapters"])
async def unload_adapter_route(adapter_identifier: str):
    """Remove an adapter from the registry."""
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    if adapter_identifier not in _manager.adapter_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter {adapter_identifier} not resident.",
        )

    _manager.unload_adapter(adapter_identifier)
    metrics.sidecar_resident_adapters.set(len(_manager.adapter_registry))
    return {"status": "success", "adapter_identifier": adapter_identifier}


@app.get("/registry/adapters", tags=["registry"])
async def get_adapters():
    """Returns the current resident adapters."""
    if _manager is None:
        return {}
    return _manager.adapter_registry


@app.post("/adapter/load/{adapter_identifier:path}", tags=["adapters"])
async def load_adapter_route(adapter_identifier: str, version: str = "latest"):
    """Trigger a LoRA adapter download (fire-and-forget, returns 202).

    Poll GET /registry/adapters to check when status becomes "loaded".
    """
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    # Already loaded — return immediately
    existing = _manager.adapter_registry.get(adapter_identifier)
    if existing and existing.get("version") == version and existing.get("status") == "loaded":
        return {"status": "loaded", "adapter_identifier": adapter_identifier, "local_path": existing["local_path"]}

    # Already downloading — don't start a second task
    if existing and existing.get("status") == "downloading":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "downloading", "adapter_identifier": adapter_identifier},
        )

    # Mark as downloading and kick off background task
    _manager.adapter_registry[adapter_identifier] = {
        "adapter_id": adapter_identifier,
        "version": version,
        "status": "downloading",
    }

    async def _background_fetch():
        try:
            start = time.time()
            local_path = await _manager.fetch_adapter(adapter_identifier, version)
            duration = time.time() - start
            metrics.sidecar_adapter_load_duration_seconds.labels(adapter=adapter_identifier).observe(duration)
            metrics.sidecar_resident_adapters.set(len(_manager.adapter_registry))
            logger.info(f"Background adapter fetch complete: {adapter_identifier} at {local_path}")
        except Exception as e:
            logger.error(f"Background adapter fetch failed for {adapter_identifier}: {e}")

    asyncio.create_task(_background_fetch())

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "downloading", "adapter_identifier": adapter_identifier},
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8")


# --- Cache endpoints (for cache-aware routing in Phase J) ---


@app.get("/cache/blocks", tags=["cache"])
async def get_cache_blocks(
    prefix_hash: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
):
    """List cached KV blocks, optionally filtered by prefix_hash and model_id."""
    if _kv_registry is None:
        return []
    if prefix_hash:
        entries = _kv_registry.query_by_prefix(prefix_hash, model_id or "")
    else:
        entries = _kv_registry.all_entries()
    return [e.to_dict() for e in entries]


@app.get("/cache/stats", tags=["cache"])
async def get_cache_stats():
    """Summary statistics for cache state (used by routing and observability)."""
    if _kv_registry is None:
        return {
            "total_blocks": 0,
            "l1_blocks": 0,
            "l2_blocks": 0,
            "l1_used_bytes": 0,
            "l2_used_bytes": 0,
            "hit_rate": 0.0,
            "eviction_count": 0,
        }
    return _kv_registry.stats()
