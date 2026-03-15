# Load-aware and prefix-cache router
# TODO: implement true load-aware routing (ie., based on queue length)


from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import httpx
from fastapi.responses import JSONResponse
from http import HTTPStatus
from pydantic import BaseModel, Field

from data_plane.gateway.config import GatewayConfig
from shared.config_loader import get_config


class GenerateRequest(BaseModel):
    model: str = Field("llama-3-8b", description="Model name from the routing table")
    prompt: str = Field("Hello, world!", description="Text prompt for generation")
    max_tokens: Optional[int] = Field(32, description="Maximum tokens to generate")


_config = GatewayConfig()
_routing_cfg = get_config("routing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: store the httpx AsyncClient instance in the application state
    app.state.http_client = httpx.AsyncClient(timeout=_config.request_timeout)
    yield
    # Shutdown: cancel the httpx client when the server shuts down
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy"}


# 2. Model-to-Service Mapping (The "Routing Table")
# In Kubernetes, this is the Service DNS name pointing to the GPU Worker Pod
MODEL_SERVICE_MAP: dict[str, str] = _routing_cfg.get("model_service_map", {
    "llama-3-8b": "http://engine:8080",
})


@app.post("/generate")
async def generate(event: GenerateRequest):
    """
    Routes the generation request to the corresponding GPU Worker Service.
    """
    model_name = event.model

    worker_url = MODEL_SERVICE_MAP.get(model_name)
    if worker_url is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    target_url = f"{worker_url}/inference"

    try:
        http_client: httpx.AsyncClient = app.state.http_client

        # We forward the entire event dictionary as the JSON body
        response = await http_client.post(
            target_url,
            json=event.model_dump(exclude_none=True)
        )

        if response.status_code >= 400:
            return JSONResponse(
                status_code=response.status_code,
                # Pass the worker's error body (decoded to string)
                content={"detail": "Worker failed to generate response.",
                         "worker_message": response.text}
            )

        return response.json()

    except httpx.ConnectError:
        # Worker is unreachable (Pod down, network issue, or not ready)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Model service '{model_name}' is unreachable."
        )
    except Exception as e:
        # Catch other unexpected errors
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during routing: {str(e)}"
        )
