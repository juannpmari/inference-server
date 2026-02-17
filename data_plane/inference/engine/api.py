from fastapi import FastAPI
from engine import Engine
from fastapi.exceptions import HTTPException
import asyncio
# from inference.kv_cache import KVCache
# kv_cache = KVCache()

app = FastAPI()


# This path is where the Sidecar puts the files (shared volume)
MODEL_PATH = "/models/resident_model"
MODEL_CONFIGS = { #each container runs only one model
    "model_name": "mistral-7b",
    "max_model_len": 4096,
}

# The cache will hold the loaded Engine objects
app.state.model_cache = None 
app.state.batching_loop = None

# Use the startup event to load ALL necessary models ONCE
@app.on_event("startup")
async def startup_event():
    """
    This runs only once when starting the container (and thus the server)
    1. Load the resident model into its engine
    2. Start the continous batching loop for the engine
    """
    print("Starting multi-model server")
    
    engine = Engine(
        model_name=MODEL_CONFIGS['model_name'],
        args={
            "max_model_len": MODEL_CONFIGS['max_model_len'],
        }
    )
    app.state.model_cache = engine

    # Start the continous batching loop
    app.state.batching_loop = asyncio.create_task(engine.continous_batching_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """ Cancel the batching loop of each engine when the server shuts down"""
    app.state.batching_loop.cancel()
    print("Continuous batching loop cancelled.")


@app.post("/inference")
async def inference(event: dict):
    """
    Generate a response for a given prompt using the resident model
    """
    prompt = event['prompt']
    # model_name = event['model']
    # max_tokens = event['max_tokens']
    # stream = event.get('stream', False)
    # lora_adapter = event['lora_adapter']

    engine = app.state.model_cache
    if engine is None:
        raise HTTPException(status_code=404, detail="Model not found")
    result = await engine.add_request(prompt=prompt)
    return result
