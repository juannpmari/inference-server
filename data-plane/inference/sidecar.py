# Sidecar
# Exposes metrics (queue depth, KV cache usage) to the controller
# Its role is to expose metrics and handle management tasks (like LoRA loading).

from inference.inference import Engine
# from inference.kv_cache import KVCache

class ArtifactManager:
    """
    An artifact manager that can load model files from disk or remote object store (S3/minio) into memory and keep metadata about which models are resident.
    """
    def __init__(self):
        pass
    
    def load(self, model_name: str, lora_adapter: str):
        pass
    
    def save(self, model_name: str, lora_adapter: str):
        pass

class Sidecar:
    def __init__(self):
        self.artifact_manager = ArtifactManager()