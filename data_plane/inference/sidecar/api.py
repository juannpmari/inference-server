import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import REGISTRY, generate_latest

from data_plane.inference.sidecar import metrics
from data_plane.inference.sidecar.artifact_manager import ArtifactManager
from data_plane.inference.sidecar.config import SidecarConfig

logger = logging.getLogger(__name__)

# Global state
_manager: Optional[ArtifactManager] = None
_config: Optional[SidecarConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _config
    logger.info("Starting sidecar...")

    _config = SidecarConfig()
    _manager = ArtifactManager(config=_config)

    # Load initial model in background
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


@app.post("/load/{model_identifier}", tags=["models"])
async def load_model_route(model_identifier: str, version: str = "latest"):
    """Load a model version and register it."""
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    try:
        start = time.time()
        local_path = await _manager.load_model(model_identifier, version)
        duration = time.time() - start

        metrics.sidecar_model_load_duration_seconds.labels(model=model_identifier, source="huggingface").observe(
            duration
        )
        metrics.sidecar_resident_models.set(len(_manager.model_registry))

        return {"status": "success", "model_identifier": model_identifier, "local_path": local_path}
    except Exception as e:
        logger.error(f"Failed to load model {model_identifier}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/unload/{model_identifier}", tags=["models"])
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


@app.get("/registry/adapters", tags=["registry"])
async def get_adapters():
    """Returns the current resident adapters."""
    if _manager is None:
        return {}
    return _manager.adapter_registry


@app.post("/adapter/fetch/{adapter_identifier}", tags=["adapters"])
async def fetch_adapter_route(adapter_identifier: str, version: str = "latest"):
    """Ensure a LoRA adapter's files are present on the shared disk."""
    if _manager is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Sidecar not initialized")

    try:
        start = time.time()
        local_path = await _manager.fetch_adapter(adapter_identifier, version)
        duration = time.time() - start

        metrics.sidecar_adapter_load_duration_seconds.labels(adapter=adapter_identifier).observe(duration)
        metrics.sidecar_resident_adapters.set(len(_manager.adapter_registry))

        return {"status": "success", "adapter_identifier": adapter_identifier, "local_path": local_path}
    except Exception as e:
        logger.error(f"Failed to fetch adapter {adapter_identifier}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch adapter: {str(e)}",
        )


@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8")
