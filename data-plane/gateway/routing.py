# Load-aware and prefix-cache router
# TODO: implement true load-aware routing (ie., based on queue length)




from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import httpx
from fastapi.responses import JSONResponse
from http import HTTPStatus

app = FastAPI()


# Use the startup event to load ALL necessary models ONCE
@app.on_event("startup")
async def startup_event():
    """
    # Store the httpx AsyncClient instance in the application state
    # Set a reasonable timeout for the *entire* request/response cycle
    """
    app.state.http_client = httpx.AsyncClient(timeout=300.0) # 5 minutes

@app.on_event("shutdown")
async def shutdown_event():
    """ Cancel the httpx client when the server shuts down"""
    await app.state.http_client.close()

# 2. Model-to-Service Mapping (The "Routing Table")
# In Kubernetes, this is the Service DNS name pointing to the GPU Worker Pod
MODEL_SERVICE_MAP = {
    "llama-3-8b": "http://vllm-llama-8b-svc:8000",
    "mistral-7b": "http://vllm-mistral-7b-svc:8000",
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service '{model_name}' is unreachable."
        )
    except Exception as e:
        # Catch other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during routing: {str(e)}"
        )