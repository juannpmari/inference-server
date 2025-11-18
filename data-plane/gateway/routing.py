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

engine = Engine(
    model_name="meta-llama/Llama-3.2-70B-Instruct",
    lora_adapter="./lora_adapter",
    args={
        "max_model_len": 8192,
        "max_batch_size": 1,
        "use_gptj": True,
        "use_gptj": True,
        "use_gptj": True,
    }
)

# kv_cache = KVCache()

@app.post("/generate")
def generate(prompt: str):
    return engine.generate(prompt)
