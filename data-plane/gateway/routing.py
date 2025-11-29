# Load-aware and prefix-cache router

# TODO: implement true load-aware routing (ie., based on queue length)
def route(model_name: str, lora_adapter: str):
    """
    Route to the correct pods based on model name and lora adapter
    The load-aware routing will be done by k8s service
    """
    pass



# API gateway (fastapi)

from fastapi import FastAPI
from inference.inference import Engine
# from inference.kv_cache import KVCache

app = FastAPI()

# kv_cache = KVCache()

# Define all models and their configurations you want to load at startup
MODEL_CONFIGS = {
    "mistral-7b": {"max_tokens": 4096},
    "llama-3-8b": {"max_tokens": 8192},
    # Add more models here...
}

# The cache will hold the loaded Engine objects
app.state.model_cache = {} 

# Use the startup event to load ALL necessary models ONCE
@app.on_event("startup")
async def startup_event():
    print("Starting multi-model server loading...")
    
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

@app.post("/generate")
def generate(event: dict):
    prompt = event['prompt']
    model_name = event['model']
    max_tokens = event['max_tokens']
    stream = event.get('stream', False)
    # lora_adapter = event['lora_adapter']

    engine = app.state.model_cache.get(model_name)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return engine.generate(prompt, max_tokens=max_tokens)
