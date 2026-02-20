# Load-aware and prefix-cache router
# TODO: implement true load-aware routing (ie., based on queue length)




from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import httpx
from fastapi.responses import JSONResponse
from http import HTTPStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: store the httpx AsyncClient instance in the application state
    # Set a reasonable timeout for the *entire* request/response cycle
    app.state.http_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes
    yield
    # Shutdown: cancel the httpx client when the server shuts down
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy"}


# 2. Model-to-Service Mapping (The "Routing Table")
# In Kubernetes, this is the Service DNS name pointing to the GPU Worker Pod
MODEL_SERVICE_MAP = {
    "llama-3-8b": "http://localhost:8080",
    # "mistral-7b": "http://vllm-mistral-7b-svc:8000",
}


@app.post("/generate")
async def generate(event: dict):
    """
    Routes the generation request to the corresponding GPU Worker Service.
    """
    model_name = event['model']

    worker_url = MODEL_SERVICE_MAP.get(model_name)
    if worker_url is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    target_url = f"{worker_url}/inference"

    try:
        http_client: httpx.AsyncClient = app.state.http_client
        
        # We forward the entire event dictionary as the JSON body
        response = await http_client.post(
            target_url, 
            json=event
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