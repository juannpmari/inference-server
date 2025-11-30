from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.utils import FlexibleArgumentParser # A custom parser for vLLM
import uuid

class Engine:
    def __init__(self, model_name: str, lora_adapter: str = None, args: dict = {}):
        self.model_name = model_name
        self.args = args
        # self.lora_adapter = lora_adapter
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
        self.request_futures:DictÑ[str,asyncio.Future] = {}

    async def add_request(self, prompt: str, sampling_params: SamplingParams = {}):
        request_id = str(self.request_counter)
        self.request_counter += 1
        
        future = asyncio.Future()
        self.request_futures[request_id] = future

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

    # def add_lora(self, lora_adapter: str):
    #     self.engine.add_lora(lora_adapter)
        