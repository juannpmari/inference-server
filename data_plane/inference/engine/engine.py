from typing import Dict, List, Any, Optional
import asyncio
import httpx
import logging

try:
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.utils import FlexibleArgumentParser
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLMEngine = None
    SamplingParams = None

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, config):
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install vllm\n"
                "Or use ENABLE_ENGINE_MOCK=true to use the mock engine."
            )

        self.config = config
        self.request_counter = 0
        self.request_futures: Dict[str, asyncio.Future] = {}

        from vllm.utils import FlexibleArgumentParser
        from vllm import EngineArgs

        parser = FlexibleArgumentParser()
        parser = EngineArgs.add_cli_args(parser)

        cli_args_list = [
            "--model", config.model_name,
            "--dtype", config.dtype,
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len)
        ]

        if config.enable_lora:
            cli_args_list.append("--enable-lora")

        args = parser.parse_args(cli_args_list)
        from vllm import EngineArgs, SamplingParams
        engine_args = EngineArgs.from_cli_args(args)
        self.sampling_params = SamplingParams(temperature=config.temperature)

        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info(f"Engine initialized with model {config.model_name}")

    async def add_request(
        self,
        prompt: str,
        adapter_identifier: Optional[str] = None,
        adapter_version: Optional[str] = None,
        sampling_params: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        if sampling_params is None:
            if temperature is not None:
                from vllm import SamplingParams
                sampling_params = SamplingParams(temperature=temperature)
            else:
                sampling_params = self.sampling_params

        request_id = str(self.request_counter)
        self.request_counter += 1

        future = asyncio.Future()
        self.request_futures[request_id] = future

        if adapter_identifier:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.config.sidecar_url}/adapter/fetch/{adapter_identifier}",
                        json={"version": adapter_version}
                    )
                    response.raise_for_status()
                    adapter = response.json()
                    adapter_path = adapter["local_path"]

                    await self._add_lora(adapter_path, adapter_identifier, adapter_version)
            except Exception as e:
                error_msg = f"Failed to load adapter {adapter_identifier} v{adapter_version}: {e}"
                logger.error(error_msg)
                future.set_exception(RuntimeError(error_msg))
                return await future

        self.engine.add_request(request_id, prompt, sampling_params)

        return await future

    def is_ready(self) -> bool:
        """Check if engine is ready to process requests"""
        return self.engine is not None

    async def continuous_batching_loop(self):
        """Main inference loop that processes batches of requests"""
        logger.info("Starting continuous batching loop...")
        try:
            while True:
                if not self.engine.has_unfinished_requests():
                    await asyncio.sleep(0.01)
                    continue

                outputs_list: List[Any] = await asyncio.to_thread(self.engine.step)

                for output in outputs_list:
                    request_id = output.request_id
                    future = self.request_futures.get(request_id)

                    if future and not future.done():
                        if output.finished:
                            response = output.outputs[0].text
                            future.set_result(response)
                            del self.request_futures[request_id]
        except asyncio.CancelledError:
            logger.info("Batching loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Batching loop error: {e}")
            raise

    
    async def _add_lora(
        self,
        adapter_path: str,
        adapter_identifier: str,
        adapter_version: str = "latest"
    ):
        from vllm.lora.request import LoRARequest
        adapter_int_id = hash(adapter_identifier + adapter_version) & 0xFFFFFFFF
        lora_request = LoRARequest(
            lora_name=f"{adapter_identifier}-{adapter_version}",
            lora_int_id=adapter_int_id,
            lora_path=adapter_path,
        )
        await asyncio.to_thread(self.engine.add_lora, lora_request)
        logger.info(f"LoRA adapter {adapter_identifier} v{adapter_version} loaded")
        