from fastapi import FastAPI
from sidecar import ArtifactManager
from fastapi.exceptions import HTTPException
from http import HTTPStatus

app = FastAPI()
manager = ArtifactManager()

async def check_initial_model_load():
    """Performs the loading of the initial model.
    This logic runs when the sidecar starts. It must be completed before the
    Inference Manager (Container B) can start its LLMEngine.
    """
    try:
        # Load a default model required by the worker
        initial_model_path = await manager.load_model(
            model_identifier="default-llama-3", 
            version="v1.0"
        )
        print(f"Initial model loaded. Path: {initial_model_path}")
        manager.is_ready = True
        
    except Exception as e:
        print(f"ERROR during initial model load: {e}")
        # Keep is_ready = False so the K8s probe fails.

# Note: We use the startup event to trigger the initial load.
@app.on_event("startup")
async def startup_event():
    import asyncio
    # Run the initial model load asynchronously in the background
    asyncio.create_task(check_initial_model_load())


@app.get("/health")
def health_check():
    """Standard Liveness Probe."""
    return {"status": "ok"}

@app.get("/ready")
def readiness_check():
    """Reports readiness only when the initial model is loaded."""
    if manager.is_ready:
        return {"status": "ready", "registry": manager.registry}
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Artifact Manager loading initial model.")

@app.post("/load/{model_identifier}")
async def load_model_route(model_identifier: str, version: str, remote_url: Optional[str] = None):
    """API to load a specific model version and update the in-memory registry."""
    
    try:
        local_path = await manager.load_model(model_identifier, version)
        return {"status": "success", "model_identifier": model_identifier, "local_path": local_path}
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load model: {str(e)}")

@app.get("/registry")
def get_registry():
    """Returns the current state of resident artifacts (models and adapters)."""
    return {"resident_models": manager.registry, "resident_adapters": manager.adapter_registry}

@app.post("/unload/{model_identifier}")
def unload_model_route(model_identifier: str):
    """Removes a model from the registry."""
    if model_identifier not in manager.registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_identifier} not resident.")
        
    manager.unload_model(model_identifier)
    return {"status": "success", "model_identifier": model_identifier}

@app.post("/adapter/fetch/{adapter_identifier}")
async def fetch_adapter_route(adapter_identifier: str, version: str):
    """
    API to ensure a LoRA adapter's delta weights are present on the shared disk.
    Called by the Inference Manager before asking vLLM to load the adapter into VRAM.
    """
    try:
        local_path = await manager.fetch_adapter(adapter_identifier, version)
        return {"status": "success", "adapter_identifier": adapter_identifier, "local_path": local_path}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch adapter: {str(e)}")
    