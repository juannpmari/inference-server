from starlette.concurrency import run_in_threadpool

# Load-aware and prefix-cache router

# TODO: implement true load-aware routing (ie., based on queue length)
def route(model_name: str, lora_adapter: str):
    """
    Route to the correct pods based on model name and lora adapter
    The load-aware routing will be done by k8s service
    """
    pass



from fastapi import FastAPI
from inference.inference import Engine
# from inference.kv_cache import KVCache

app = FastAPI()

# kv_cache = KVCache()

MODEL_CONFIGS = {
    "mistral-7b": {"max_tokens": 4096},
    "llama-3-8b": {"max_tokens": 8192},
    # Add more models
}

# The cache will hold the loaded Engine objects
app.state.model_cache = {} 
app.state.batching_loops = []

# Use the startup event to load ALL necessary models ONCE
@app.on_event("startup")
async def startup_event():
    """
    1. Load all models into their engines
    2. Start the continous batching loop for each engine
    """
    print("Starting multi-model server")
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"Loading model: {model_name}...")
        engine = Engine(
            model_name=model_name,
            args={
                "max_model_len": config['max_tokens'],
            }
        )
        app.state.model_cache[model_name] = engine
    print("All configured models loaded successfully.")

    # Start the continous batching loop for each engine
    for _, engine in app.state.model_cache.items():
        app.state.batching_loops.append(asyncio.create_task(engine.continous_batching_loop()))

@app.on_event("shutdown")
async def shutdown_event():
    """ Cancel the batching loop of each engine when the server shuts down"""
    for loop in app.state.batching_loops:
        loop.cancel()
        print("Continuous batching loop cancelled.")


@app.post("/generate")
async def generate(event: dict):
    """
    Generate a response for a given prompt using the specified model
    """
    prompt = event['prompt']
    model_name = event['model']
    # max_tokens = event['max_tokens']
    # stream = event.get('stream', False)
    # lora_adapter = event['lora_adapter']

    engine = app.state.model_cache.get(model_name)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    result = await engine.add_request(prompt=prompt)

    return result
