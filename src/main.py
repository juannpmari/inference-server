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
