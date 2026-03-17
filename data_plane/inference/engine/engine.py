from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
import time

try:
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.engine.arg_utils import FlexibleArgumentParser
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLMEngine = None
    SamplingParams = None

from data_plane.inference.engine import metrics
from data_plane.inference.engine.lora_manager import LoRAManager
from shared.monitoring.models import TimingInfo, RequestRecord

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, config, model_path: str = None, collector=None):
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install vllm\n"
                "Or use ENABLE_ENGINE_MOCK=true to use the mock engine."
            )

        self.config = config
        self.request_counter = 0
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.request_timings: Dict[str, TimingInfo] = {}
        self.collector = collector

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

    def tokenize(self, text: str) -> list:
        """Tokenize text using the engine's tokenizer."""
        return self.engine.get_tokenizer().encode(text)

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

        # Create timing info
        timing = TimingInfo(
            submitted_at=time.time(),
            adapter_id=adapter_identifier,
            input_tokens=len(self.tokenize(prompt)),
        )

        lora_request = None
        if adapter_identifier and self.lora_manager:
            try:
                lora_request, swap_duration = await self.lora_manager.ensure_adapter_loaded(
                    adapter_identifier=adapter_identifier,
                    adapter_version=adapter_version,
                )
                timing.adapter_swap_latency_s = swap_duration
            except Exception as e:
                error_msg = f"Failed to load adapter {adapter_identifier} v{adapter_version}: {e}"
                logger.error(error_msg)
                future.set_exception(RuntimeError(error_msg))
                return await future

        self.request_timings[request_id] = timing

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

                # Record batch size
                if self.collector and outputs_list:
                    self.collector.record_batch_size(len(outputs_list))
                    metrics.engine_batch_size.labels(model=self.config.model_name).observe(len(outputs_list))

                for output in outputs_list:
                    request_id = output.request_id
                    timing = self.request_timings.get(request_id)
                    future = self.request_futures.get(request_id)

                    # Update timing
                    if timing:
                        now = time.time()
                        if timing.processing_started_at == 0.0:
                            timing.processing_started_at = now
                        if timing.first_token_at == 0.0 and output.outputs[0].text:
                            timing.first_token_at = now
                        timing.last_step_at = now
                        timing.step_count += 1

                    if future and not future.done():
                        if output.finished:
                            response = output.outputs[0].text
                            future.set_result(response)
                            del self.request_futures[request_id]

                            # Finalize timing and record
                            if timing:
                                timing.finished_at = time.time()
                                timing.output_tokens = timing.step_count
                                # Try to get accurate input token count from vLLM output
                                if hasattr(output, 'prompt_token_ids') and output.prompt_token_ids:
                                    timing.input_tokens = len(output.prompt_token_ids)
                                record = RequestRecord.from_timing(request_id, self.config.model_name, timing)
                                if self.collector:
                                    self.collector.record_request(record)
                                # Observe Prometheus metrics
                                model = self.config.model_name
                                if record.ttft_s > 0:
                                    metrics.engine_time_to_first_token_seconds.labels(model=model).observe(record.ttft_s)
                                if record.queue_wait_s > 0:
                                    metrics.engine_queue_wait_seconds.labels(model=model).observe(record.queue_wait_s)
                                if record.prefill_s > 0:
                                    metrics.engine_prefill_seconds.labels(model=model).observe(record.prefill_s)
                                if record.inter_token_latency_s > 0:
                                    metrics.engine_inter_token_latency_seconds.labels(model=model).observe(record.inter_token_latency_s)
                                if record.input_tokens > 0:
                                    metrics.engine_input_tokens_per_request.labels(model=model).observe(record.input_tokens)
                                if record.output_tokens > 0:
                                    metrics.engine_output_tokens_per_request.labels(model=model).observe(record.output_tokens)
                                if record.tokens_per_second > 0:
                                    metrics.engine_decode_tokens_per_second.labels(model=model).observe(record.tokens_per_second)
                                    metrics.engine_tokens_per_second.labels(model=model).set(record.tokens_per_second)
                                del self.request_timings[request_id]

        except asyncio.CancelledError:
            logger.info("Batching loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Batching loop error: {e}")
            raise
