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
        self.engine =  LLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=0.0)

    def _generate_request_id(self):
        return str(uuid.uuid4())

    def generate(self, prompt: str, **opts):
        request_id = self._generate_request_id()
        self.engine.add_request(
            request_id,
            prompt,
            self.sampling_params
        )

        for i in range(opts.get("max_tokens", 10)):
            result = self.engine.step()
            if result is not None:
                return result

    def generate_stream(self, prompt: str, **opts):
        yield from self.engine.generate_stream(prompt, **opts)

    def add_lora(self, lora_adapter: str):
        self.engine.add_lora(lora_adapter)

    def batch_requests(self, prompts: list[str], **opts):
        return self.engine.batch_requests(prompts, **opts)
        