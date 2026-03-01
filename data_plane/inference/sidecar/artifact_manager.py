import asyncio
import logging
import os
import shutil
import tempfile
from typing import Dict, Optional

from huggingface_hub import snapshot_download

from data_plane.inference.sidecar.config import SidecarConfig

logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    An artifact manager that can load model files from disk or HuggingFace Hub
    into a shared volume and keep metadata about which models are resident.
    """

    def __init__(self, config: Optional[SidecarConfig] = None):
        self.config = config or SidecarConfig()

        # Base Models (Key: model_identifier, Value: metadata dict)
        self.model_registry: Dict[str, Dict] = {}

        # LoRA Adapters (Key: adapter_identifier, Value: metadata dict)
        self.adapter_registry: Dict[str, Dict] = {}

        self.is_ready = False
        self.max_resident_adapters = self.config.max_adapters

        # Ensure the shared volume path exists
        os.makedirs(self.config.shared_volume, exist_ok=True)
        logger.info(f"Artifact Manager initialized. Storage path: {self.config.shared_volume}")

        # Restore registry from disk if available
        self._load_registry()

    def _load_registry(self):
        """Restore registry state from JSON file on disk."""
        import json

        if os.path.exists(self.config.registry_path):
            try:
                with open(self.config.registry_path) as f:
                    data = json.load(f)
                self.model_registry = data.get("models", {})
                self.adapter_registry = data.get("adapters", {})
                logger.info(
                    f"Registry restored: {len(self.model_registry)} models, {len(self.adapter_registry)} adapters"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not restore registry: {e}")

    def _persist_registry(self):
        """Persist registry state to JSON file on disk."""
        import json

        try:
            os.makedirs(os.path.dirname(self.config.registry_path), exist_ok=True)
            with open(self.config.registry_path, "w") as f:
                json.dump({"models": self.model_registry, "adapters": self.adapter_registry}, f, indent=2)
        except OSError as e:
            logger.error(f"Could not persist registry: {e}")

    async def _fetch_from_external_storage(self, artifact_type: str, identifier: str, version: str) -> str:
        """
        Download model/adapter files from HuggingFace Hub.

        Uses atomic download: downloads to a temp directory, then renames into place.
        """
        local_target_path = os.path.join(self.config.shared_volume, identifier.replace("/", "--"), version)

        if os.path.exists(local_target_path) and os.listdir(local_target_path):
            logger.info(f"{artifact_type} {identifier} v{version} already cached at {local_target_path}")
            return local_target_path

        logger.info(f"Downloading {artifact_type} {identifier} v{version} from HuggingFace Hub...")

        # Atomic download: use temp dir, then rename
        tmp_dir = tempfile.mkdtemp(dir=self.config.shared_volume, prefix=f".dl-{artifact_type}-")
        try:
            await asyncio.to_thread(
                snapshot_download,
                repo_id=identifier,
                revision=version if version != "latest" else None,
                local_dir=tmp_dir,
            )

            # Atomic rename into final location
            os.makedirs(os.path.dirname(local_target_path), exist_ok=True)
            if os.path.exists(local_target_path):
                shutil.rmtree(local_target_path)
            shutil.move(tmp_dir, local_target_path)

            logger.info(f"Download complete. {artifact_type} files ready at: {local_target_path}")
            return local_target_path

        except Exception:
            # Clean up temp dir on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    async def load_model(self, model_identifier: str, version: str = "latest") -> str:
        """Downloads the model if necessary and registers it as resident."""

        existing = self.model_registry.get(model_identifier)
        if existing and existing.get("version") == version and existing.get("status") == "loaded":
            logger.info(f"Model {model_identifier} v{version} already resident.")
            return existing["local_path"]

        local_path = await self._fetch_from_external_storage("model", model_identifier, version)
        self.model_registry[model_identifier] = {
            "model_id": model_identifier,
            "version": version,
            "local_path": local_path,
            "status": "loaded",
        }
        self._persist_registry()
        return local_path

    def unload_model(self, model_identifier: str):
        """Removes the model from the registry."""
        if model_identifier in self.model_registry:
            del self.model_registry[model_identifier]
            self._persist_registry()
            logger.info(f"Model {model_identifier} unloaded from registry.")

    def unload_adapter(self, adapter_identifier: str):
        """Removes the adapter from the registry."""
        if adapter_identifier in self.adapter_registry:
            del self.adapter_registry[adapter_identifier]
            self._persist_registry()
            logger.info(f"Adapter {adapter_identifier} unloaded from registry.")

    async def fetch_adapter(self, adapter_identifier: str, version: str = "latest") -> str:
        """Downloads the adapter if necessary and registers it as resident.

        Updates adapter_registry status: "downloading" -> "loaded" (or removes on failure).
        Caller should set status to "downloading" before calling if using fire-and-forget pattern.
        """
        existing = self.adapter_registry.get(adapter_identifier)
        if existing and existing.get("version") == version and existing.get("status") == "loaded":
            logger.info(f"Adapter {adapter_identifier} v{version} already resident.")
            return existing["local_path"]

        try:
            local_path = await self._fetch_from_external_storage("adapter", adapter_identifier, version)
            self.adapter_registry[adapter_identifier] = {
                "adapter_id": adapter_identifier,
                "version": version,
                "local_path": local_path,
                "status": "loaded",
            }
            self._persist_registry()
            return local_path
        except Exception:
            # Remove failed entry so it can be retried
            self.adapter_registry.pop(adapter_identifier, None)
            self._persist_registry()
            raise
