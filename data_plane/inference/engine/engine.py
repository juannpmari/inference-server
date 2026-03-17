from typing import Dict, List, Any, Optional
import asyncio
import json
import logging

try:
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.engine.arg_utils import FlexibleArgumentParser
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLMEngine = None
    SamplingParams = None

from data_plane.inference.engine.lora_manager import LoRAManager

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, config, model_path: str = None):
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install vllm\n"
                "Or use ENABLE_ENGINE_MOCK=true to use the mock engine."
            )

        self.config = config
        self.request_counter = 0
        self.request_futures: Dict[str, asyncio.Future] = {}

        parser = FlexibleArgumentParser()
        parser = EngineArgs.add_cli_args(parser)

        resolved_model = model_path or config.model_name
        cli_args_list = [
            "--model", resolved_model,
            "--dtype", config.dtype,
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len)
        ]

        if config.enable_lora:
            cli_args_list.extend([
                "--enable-lora",
                "--max-loras", str(config.max_loras),
                "--max-lora-rank", str(config.max_lora_rank),
            ])

        if config.enable_kv_offload:
            cli_args_list.extend([
                "--kv-transfer-config", json.dumps({
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "spec_name": "SidecarOffloadingSpec",
                        "spec_module_path": "data_plane.inference.engine.kv_offload.sidecar_spec",
                        "sidecar_grpc_url": config.sidecar_grpc_url,
                        "num_blocks": config.kv_offload_num_blocks,
                    },
                }),
            ])

        args = parser.parse_args(cli_args_list)
        engine_args = EngineArgs.from_cli_args(args)
        self.sampling_params = SamplingParams(temperature=config.temperature)

        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info(f"Engine initialized with model {resolved_model}")

        self.lora_manager = None
        if config.enable_lora:
            self.lora_manager = LoRAManager(
                engine=self.engine,
                config=config,
                sidecar_url=config.sidecar_url,
            )

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
            from vllm import SamplingParams
            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            else:
                kwargs["temperature"] = self.config.temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            sampling_params = SamplingParams(**kwargs)

        request_id = str(self.request_counter)
        self.request_counter += 1

        future = asyncio.Future()
        self.request_futures[request_id] = future

        lora_request = None
        if adapter_identifier and self.lora_manager:
            try:
                lora_request = await self.lora_manager.ensure_adapter_loaded(
                    adapter_identifier=adapter_identifier,
                    adapter_version=adapter_version,
                )
            except Exception as e:
                error_msg = f"Failed to load adapter {adapter_identifier} v{adapter_version}: {e}"
                logger.error(error_msg)
                future.set_exception(RuntimeError(error_msg))
                return await future

        if lora_request:                                                                                                                      
            logger.info(f"Submitting request {request_id} WITH adapter: {lora_request.lora_name} (id={lora_request.lora_int_id}, path={lora_request.lora_path})")                                                                                                      
        else:           
            logger.info(f"Submitting request {request_id} with base model only")
            
        self.engine.add_request(request_id, prompt, sampling_params, lora_request=lora_request)

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

