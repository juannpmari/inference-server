# Sidecar
# Exposes metrics (queue depth, KV cache usage) to the controller
# Its role is to expose metrics and handle management tasks (like LoRA loading).

# from inference.kv_cache import KVCache

# The shared volume mount point where models will be stored to be served by the inference engine
SHARED_VOLUME_PATH = "/mnt/models"

class ArtifactManager:
    """
    An artifact manager that can load model files from disk or remote object store (S3) into memory and keep metadata about which models are resident.
    """
    def __init__(self):
        # Base Models (Key: model_identifier, Value: Local path)
        self.model_registry: Dict[str, str] = {}

        # LoRA Adapters (Key: adapter_identifier, Value: Local path)
        self.adapter_registry: Dict[str, Dict] = {}
        
        self.is_ready = False
        self.MAX_RESIDENT_ADAPTERS = 10 # <-- VRAM constraint
        
        # Ensure the shared volume path exists
        os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)
        print(f"Artifact Manager initialized. Storage path: {SHARED_VOLUME_PATH}")

    async def _fetch_from_external_storage(self, artifact_type: str, identifier: str, version: str):
        """
        Download model files from remote storage (S3)
        Args:
            artifact_type: The type of artifact to download (e.g., "model", "adapter")
            identifier: The identifier of the artifact
            version: The version of the artifact
        Returns:
            The local path where the artifact was successfully downloaded.
        """
        local_target_path = os.path.join(SHARED_VOLUME_PATH, identifier, version)
        if os.path.exists(local_target_path):
            print(f"{artifact_type} {identifier} v{version} already cached.")
            return local_target_path

        os.makedirs(local_target_path, exist_ok=True)
        # Simulate large file download
        await asyncio.sleep(5) 
        print(f"Download complete. Model files ready at: {local_target_path}")
        return local_target_path
    
    async def load_model(self, model_identifier: str, version: str) -> str:
        """Downloads the model if necessary and registers it as resident."""
        
        if self.model_registry.get(model_identifier) and self.model_registry[model_identifier].endswith(version):
            print(f"Model {model_identifier} v{version} already resident.")
            return self.model_registry[model_identifier]

        local_path = await self._fetch_from_external_storage("model", model_identifier, version)
        self.model_registry[model_identifier] = local_path
        return local_path

    def unload_model(self, model_identifier: str): #CHECK: does this unload the model from gpu vram?
        """Removes the model from the registry and optionally cleans up files."""
        if model_identifier in self.model_registry:
            del self.model_registry[model_identifier]
            # NOTE: For complex systems, you might only remove the registration 
            # but leave files for caching purposes.
            # shutil.rmtree(os.path.join(SHARED_VOLUME_PATH, model_identifier), ignore_errors=True)
            print(f"Model {model_identifier} unloaded from registry.")

    async def fetch_adapter(self, adapter_identifier: str, version: str) -> str:
        """Downloads the adapter if necessary and registers it as resident. This is called by the Inference Manager to ensure the file is on disk."""
        local_path = await self._fetch_from_external_storage("adapter", adapter_identifier, version)
        self.adapter_registry[adapter_identifier] = local_path
        return local_path
        # NOTE: No unload_adapter needed. Files stay on disk for caching, and vLLM handles the VRAM eviction automatically.
        