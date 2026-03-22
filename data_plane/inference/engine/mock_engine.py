"""
Mock LLM Engine for testing without GPU.
Provides a deterministic mock implementation with the same interface as the real Engine.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from data_plane.inference.engine import metrics
from data_plane.inference.engine.sidecar_cache_client import SidecarCacheClient
from shared.monitoring.models import TimingInfo, RequestRecord

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

    def __init__(self, config=None, collector=None):
        self.config = config
        self.request_counter = 0
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.request_timings: Dict[str, TimingInfo] = {}
        self.request_queues: Dict[str, asyncio.Queue] = {}
        self.pending_requests = {}
        self.finished_requests = []
        self.collector = collector

        # LoRA adapter tracking
        self.loaded_loras: Dict[int, object] = {}
        self.lora_load_count: int = 0
        self.lora_remove_count: int = 0

        # Cache client for KV block offload/fetch (created from config or set later)
        self.cache_client: Optional[SidecarCacheClient] = None
        if config and getattr(config, "sidecar_grpc_url", None):
            self.cache_client = SidecarCacheClient(
                grpc_url=config.sidecar_grpc_url,
            )

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

    def tokenize(self, text: str) -> list:
        """Approximate token count: ~4 chars per token."""
        return list(range(len(text) // 4 or 1))

    async def add_request(
        self,
        prompt: str,
        adapter_identifier: Optional[str] = None,
        adapter_version: Optional[str] = None,
        sampling_params=None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Add a request to the mock engine"""
        request_id = str(self.request_counter)
        self.request_counter += 1

        future = asyncio.Future()
        self.request_futures[request_id] = future

        # Create timing info
        timing = TimingInfo(
            submitted_at=time.time(),
            adapter_id=adapter_identifier,
            input_tokens=len(self.tokenize(prompt)),
        )

        # Handle LoRA adapter (same pattern as real Engine)
        if adapter_identifier and self.lora_manager:
            try:
                _, swap_duration = await self.lora_manager.ensure_adapter_loaded(
                    adapter_identifier=adapter_identifier,
                    adapter_version=adapter_version,
                )
                timing.adapter_swap_latency_s = swap_duration
            except Exception as e:
                future.set_exception(RuntimeError(f"Failed to load adapter: {e}"))
                return await future

        self.request_timings[request_id] = timing

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

    async def add_streaming_request(
        self,
        prompt: str,
        adapter_identifier: Optional[str] = None,
        adapter_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> asyncio.Queue:
        """Return an asyncio.Queue that yields mock token deltas then a None sentinel."""
        queue: asyncio.Queue = asyncio.Queue()
        response_text = self._generate_mock_response(prompt)
        input_tokens = len(self.tokenize(prompt))

        async def _produce():
            words = response_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                await queue.put({"token": token, "finish_reason": None})
                await asyncio.sleep(0.001)
            await queue.put({
                "token": "",
                "finish_reason": "stop",
                "prompt_tokens": input_tokens,
                "completion_tokens": len(words),
            })
            await queue.put(None)

        asyncio.create_task(_produce())
        return queue

    def apply_chat_template(self, messages: list, add_generation_prompt: bool = True) -> str:
        """Simple concatenation fallback for mock engine."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

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
        Records timing data for the monitoring system.
        """
        outputs = []
        now = time.time()

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

            # Record timing and build RequestRecord
            timing = self.request_timings.get(request_id)
            if timing:
                timing.processing_started_at = timing.submitted_at + 0.001
                timing.first_token_at = timing.submitted_at + 0.002
                timing.last_step_at = now
                timing.finished_at = now
                timing.step_count = len(self.tokenize(request_data["response"]))
                timing.output_tokens = timing.step_count

                record = RequestRecord.from_timing(request_id, self.config.model_name if self.config else "mock", timing)
                if self.collector:
                    self.collector.record_request(record)

                # Observe Prometheus metrics
                model = self.config.model_name if self.config else "mock"
                if record.ttft_s > 0:
                    metrics.engine_time_to_first_token_seconds.labels(model=model).observe(record.ttft_s)
                if record.tokens_per_second > 0:
                    metrics.engine_tokens_per_second.labels(model=model).set(record.tokens_per_second)

                del self.request_timings[request_id]

            # Fulfill the future
            if request_id in self.request_futures:
                future = self.request_futures[request_id]
                if not future.done():
                    future.set_result(request_data["response"])
                    del self.request_futures[request_id]

        # Record batch size
        if self.collector and outputs:
            self.collector.record_batch_size(len(outputs))
            model = self.config.model_name if self.config else "mock"
            metrics.engine_batch_size.labels(model=model).observe(len(outputs))

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

    async def offload_block(
        self,
        block_hash: str,
        data: bytes,
        model_id: str = "",
    ) -> bool:
        """Simulate what vLLM's OffloadingConnector would do: offload a KV block via gRPC."""
        if not self.cache_client:
            return False

        # Allocate a slot, store data
        ids = await self.cache_client.allocate_blocks([block_hash])
        if ids is None:
            return False
        block_id = ids[0]
        return await self.cache_client.store_block(
            block_id=block_id,
            block_hash=block_hash,
            data=data,
            model_id=model_id,
        )

    async def fetch_block(self, block_hash: str, block_id: int) -> Optional[bytes]:
        """Fetch a cached KV block back via gRPC."""
        if not self.cache_client:
            return None
        return await self.cache_client.load_block(block_id)

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
