from abc import abstractmethod

import ray
import torch

from vllm import LLM, SamplingParams


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, premises):
        return


torch.set_float32_matmul_precision("medium")


def get_prev_tactics(goal):
    if not goal.in_edges:
        return ''
    else:
        return get_prev_tactics(goal.in_edges[0].src) + goal.in_edges[0].tactic


class InternLMGenerator:
    def __init__(self, config) -> None:
        self.sampling_params = SamplingParams(**config.sampling_params)
        self.llm = LLM(**config.model_params)

    def generate(self, state, retriever_args, num_samples):
        outputs = self.llm.generate(state, self.sampling_params, use_tqdm=False)

        outputs = [(i.text.strip(), i.cumulative_logprob) for i in outputs[0].outputs]

        outputs = list(set(outputs))

        outputs = sorted(outputs, key=lambda x: x[1], reverse=True)

        return outputs, state


class InternLMWrapper(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goal, premises):
        path, theorem, position = premises

        state = f"---\nNAME: {theorem.full_name}\n\n---\nPROOF_BEFORE: {get_prev_tactics(goal)}\n\n---\nSTATE_BEFORE: {goal.goal}\n\n---\nTACTIC: "

        tactics = ray.get(self.tac_model.get_tactics.remote(state, premises))

        return tactics
