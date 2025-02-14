import torch

from vllm import LLM, SamplingParams

torch.set_float32_matmul_precision("medium")


# todo training, etc. from GenTacModel

class InternLMGenerator:
    def __init__(self, config) -> None:
        self.sampling_params = SamplingParams(**config.sampling_params)
        self.llm = LLM(**config.model_params)

    def generate(self, state, retriever_args, num_samples):
        outputs = self.llm.generate(state, self.sampling_params)

        outputs = [(i.text.strip(), i.cumulative_logprob) for i in outputs[0].outputs]

        return outputs, state

