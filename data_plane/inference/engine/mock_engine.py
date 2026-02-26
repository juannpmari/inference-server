"""
Mock LLM Engine for testing without GPU.
Provides a deterministic mock implementation with the same interface as the real Engine.
"""

import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Deterministic mock responses based on prompt
MOCK_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "who": "I'm an AI assistant created by Anthropic.",
    "what": "I'm a language model trained to be helpful, harmless, and honest.",
    "test": "This is a test response from the mock engine.",
}

DEFAULT_MOCK_RESPONSE = "The quick brown fox jumps over the lazy dog. This is a mock response."


class MockLLMOutput:
    """Mock output object that mimics vLLM output format"""

    def __init__(self, request_id: str, text: str, finished: bool = True):
        self.request_id = request_id
        self.finished = finished
        self.outputs = [MockOutputToken(text)]


class MockOutputToken:
    """Mock output token that mimics vLLM output format"""

    def __init__(self, text: str):
        self.text = text


class MockLLMEngine:
    """
    Mock LLM Engine that simulates vLLM behavior without requiring a GPU.
    Used for testing and development on machines without GPU support.
    """

    def __init__(self, config=None):
        self.config = config
        self.request_counter = 0
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.pending_requests = {}
        self.finished_requests = []

        # LoRA adapter tracking
        self.loaded_loras: Dict[int, object] = {}
        self.lora_load_count: int = 0
        self.lora_remove_count: int = 0

        # Initialize LoRA manager if enabled
        self.lora_manager = None
        if config and getattr(config, "enable_lora", False):
            from data_plane.inference.engine.lora_manager import LoRAManager
            self.lora_manager = LoRAManager(
                engine=self,
                config=config,
                sidecar_url=getattr(config, "sidecar_url", "http://localhost:8001"),
            )

        logger.info("MockLLMEngine initialized")

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return True

    async def add_request(
        self,
        prompt: str,
        adapter_identifier: Optional[str] = None,
        adapter_version: Optional[str] = None,
        sampling_params=None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Add a request to the mock engine"""
        request_id = str(self.request_counter)
        self.request_counter += 1

        future = asyncio.Future()
        self.request_futures[request_id] = future

        # Handle LoRA adapter (same pattern as real Engine)
        if adapter_identifier and self.lora_manager:
            try:
                await self.lora_manager.ensure_adapter_loaded(
                    adapter_identifier=adapter_identifier,
                    adapter_version=adapter_version,
                )
            except Exception as e:
                future.set_exception(RuntimeError(f"Failed to load adapter: {e}"))
                return await future

        # Generate deterministic response based on prompt
        response_text = self._generate_mock_response(prompt)

        # Store in pending for the batching loop to process
        self.pending_requests[request_id] = {
            "prompt": prompt,
            "response": response_text,
            "adapter": adapter_identifier
        }

        # Schedule immediate resolution so add_request doesn't hang
        # when no batching loop is running (e.g., in tests)
        asyncio.get_running_loop().call_soon(self._resolve_pending)

        return await future

    def _resolve_pending(self):
        """Resolve all pending request futures immediately."""
        for request_id, request_data in list(self.pending_requests.items()):
            if request_id in self.request_futures:
                future = self.request_futures[request_id]
                if not future.done():
                    future.set_result(request_data["response"])
                    del self.request_futures[request_id]
            del self.pending_requests[request_id]

    def has_unfinished_requests(self) -> bool:
        """Check if there are pending requests"""
        return len(self.pending_requests) > 0

    async def step(self):
        """
        Simulate a step of the inference loop.
        In mock mode, immediately return finished requests.
        """
        outputs = []

        # Process all pending requests
        for request_id, request_data in list(self.pending_requests.items()):
            output = MockLLMOutput(
                request_id=request_id,
                text=request_data["response"],
                finished=True
            )
            outputs.append(output)
            self.finished_requests.append(request_id)
            del self.pending_requests[request_id]

            # Fulfill the future
            if request_id in self.request_futures:
                future = self.request_futures[request_id]
                if not future.done():
                    future.set_result(request_data["response"])
                    del self.request_futures[request_id]

        return outputs

    async def continuous_batching_loop(self):
        """Main batching loop for mock engine"""
        logger.info("Starting mock continuous batching loop...")
        try:
            while True:
                if not self.has_unfinished_requests():
                    await asyncio.sleep(0.01)
                    continue

                await self.step()
                # Small delay to simulate GPU processing
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Mock batching loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Mock batching loop error: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        """Synchronous generation method for simple usage."""
        return self._generate_mock_response(prompt)

    def add_lora(self, lora_request):
        """Mock LoRA loading — tracks adapter in internal state."""
        int_id = lora_request.lora_int_id
        name = lora_request.lora_name
        self.loaded_loras[int_id] = lora_request
        self.lora_load_count += 1
        logger.info(f"Mock: LoRA adapter {name} (id={int_id}) loaded. "
                     f"Total loaded: {len(self.loaded_loras)}")

    def remove_lora(self, lora_int_id: int):
        """Mock LoRA removal — removes adapter from internal state."""
        removed = self.loaded_loras.pop(lora_int_id, None)
        self.lora_remove_count += 1
        name = getattr(removed, "lora_name", str(lora_int_id)) if removed else str(lora_int_id)
        logger.info(f"Mock: LoRA adapter {name} (id={lora_int_id}) removed. "
                     f"Total loaded: {len(self.loaded_loras)}")

    @staticmethod
    def _generate_mock_response(prompt: str) -> str:
        """Generate a deterministic mock response based on prompt"""
        prompt_lower = prompt.lower()

        # Check for keywords in the prompt
        for keyword, response in MOCK_RESPONSES.items():
            if keyword in prompt_lower:
                return response

        # Return default response
        return DEFAULT_MOCK_RESPONSE
