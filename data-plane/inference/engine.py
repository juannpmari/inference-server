# LLMEngine
from vllm import LLMEngine

class Engine:
    def __init__(self, model_name: str, lora_adapter: str, args: dict):
        self.model_name = model_name
        self.lora_adapter = lora_adapter
        self.engine = LLMEngine(model_name, lora_adapter, **args)

    def generate(self, prompt: str, **kwargs):
        return self.engine.generate(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        yield from self.engine.generate_stream(prompt, **kwargs)

    def add_lora(self, lora_adapter: str):
        self.engine.add_lora(lora_adapter)

    def batch_requests(self, prompts: list[str], **kwargs):
        return self.engine.batch_requests(prompts, **kwargs)
        