# Sidecar
# Exposes metrics (queue depth, KV cache usage) to the controller
# Its role is to expose metrics and handle management tasks (like LoRA loading).

from inference.inference import Engine
# from inference.kv_cache import KVCache

# The shared volume mount point where models will be stored to be served by the inference engine
SHARED_VOLUME_PATH = "/mnt/models"

class ArtifactManager:
    """
    An artifact manager that can load model files from disk or remote object store (S3) into memory and keep metadata about which models are resident.
    """
    def __init__(self):
        # Dictionary to track which models are currently loaded/resident
        # Key: model_identifier (e.g., "llama-3-8b")
        # Value: Local filesystem path where the model is ready
        self.registry: Dict[str, str] = {}
        self.is_ready = False
        
        # Ensure the shared volume path exists
        os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)
        print(f"Artifact Manager initialized. Storage path: {SHARED_VOLUME_PATH}")
    
    async def download_model(self, model_identifier: str, version: str) -> str:
        """
        MOCK function for fetching model files from remote storage (S3/MinIO).
        This would use boto3 or MinIO client in a real scenario.
        
        Returns: The local path where the model was successfully downloaded.
        """
        print(f"Initiating download for {model_identifier} v{version}...")
        
        # 1. Define target path
        local_target_path = os.path.join(SHARED_VOLUME_PATH, model_identifier, version)
        
        # 2. MOCK: Create dummy directory and files
        os.makedirs(local_target_path, exist_ok=True)
        # Simulate large file download
        await asyncio.sleep(5) 
        print(f"Download complete. Model files ready at: {local_target_path}")
        
        return local_target_path

    async def load_model(self, model_identifier: str, version: str, remote_url: Optional[str] = None) -> str:
        """Downloads the model if necessary and registers it as resident."""
        
        # Check if model is already loaded and current version is desired
        if self.registry.get(model_identifier) and self.registry[model_identifier].endswith(version):
            print(f"Model {model_identifier} v{version} already resident.")
            return self.registry[model_identifier]

        # 1. Download/Cache the model files
        local_path = await self.download_model(model_identifier, version)
        
        # 2. Update the in-memory registry
        self.registry[model_identifier] = local_path
        
        return local_path

    def unload_model(self, model_identifier: str):
        """Removes the model from the registry and optionally cleans up files."""
        if model_identifier in self.registry:
            del self.registry[model_identifier]
            # NOTE: For complex systems, you might only remove the registration 
            # but leave files for caching purposes.
            # shutil.rmtree(os.path.join(SHARED_VOLUME_PATH, model_identifier), ignore_errors=True)
            print(f"Model {model_identifier} unloaded from registry.")