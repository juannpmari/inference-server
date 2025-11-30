from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.utils import FlexibleArgumentParser # A custom parser for vLLM
from vllm.lora.request import LoRARequest
from typing import Dict, List, Any
import asyncio
import requests


SIDECAR_BASE_URL = "http://localhost:8001"

class Engine:
    def __init__(self, model_name: str, args: dict = {}):
        self.model_name = model_name
        self.args = args
        parser = FlexibleArgumentParser()
        parser = EngineArgs.add_cli_args(parser)

        cli_args_list = [
            "--model", self.model_name,
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.80",
            "--max_model_len", self.args.get("max_model_len", 10)
        ]
        args = parser.parse_args(cli_args_list)
        engine_args = EngineArgs.from_cli_args(args)
        self.sampling_params = SamplingParams(temperature=0.0)
        
        self.engine =  LLMEngine.from_engine_args(engine_args)
        self.request_counter = 0
        self.request_futures:Dict[str,asyncio.Future] = {}

    async def add_request(self, prompt: str, adapter_identifier:str = None, adapter_version:str = None, sampling_params: SamplingParams = {}):
        request_id = str(self.request_counter)
        self.request_counter += 1
        
        future = asyncio.Future()
        self.request_futures[request_id] = future

        # TODO: offload this to a cache so that subsequent requests don't go through the latency of connecting to the sidecar (Cache-and-Wait Synchronization)
        if adapter_identifier:
            try:
                response = asyncio.to_thread(
                    requests.post,
                    f"{SIDECAR_BASE_URL}/adapter/fetch/{adapter_identifier}",
                    json={"version": adapter_version}
                )
                response.raise_for_status()
                adapter = response.json()
                adapter_path = adapter["local_path"]
                
                await self._add_lora(adapter_path, adapter_identifier, adapter_version)
            except Exception as e:
                future.set_exception(RuntimeError(f"Failed to load adapter {adapter_identifier} v{adapter_version}: {e}"))
                return await future

        self.engine.add_request(
            request_id,
            prompt,
            self.sampling_params
        )

        return await future

    async def continous_batching_loop(self):
        print("Starting continous batching loop...")
        while True:
            if not self.engine.has_unfinished_requests(): #Nothing to process
                await asyncio.sleep(0.01)
                continue
            
            outputs_list: List[Any] = await asyncio.to_thread(self.engine.step) #generates one token for each request

            for output in outputs_list:
                request_id = output.request_id
                future = self.request_futures.get(request_id)
                
                if future and not future.done():
                    if output.finished: #Request reached EOS token or max_tokens so it's finished
                        response = output.outputs[0].text
                        future.set_result(response)
                        del self.request_futures[request_id]
                        # NOTE: For streaming, send partial results here

    # def generate_stream(self, prompt: str, **opts):
    #     yield from self.engine.generate_stream(prompt, **opts)

    
    # TODO: complete this connection to add required adapters on demand 
    async def _add_lora(self, adapter_path: str, adapter_identifier: str, adapter_version: str = "latest"):
        adapter_path = "/path/to/lora/files" #TODO: get from sidecar
        adapter_int_id = hash(adapter_identifier + adapter_version) & 0xFFFFFFFF
        request = LoRARequest(
            lora_name=f"{adapter_identifier}-{adapter_version}",
            lora_int_id=adapter_int_id,
            lora_path=adapter_path,
        )
        await self.engine.add_lora(request)
        