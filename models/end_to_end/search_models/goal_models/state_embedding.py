# import pytorch_lightning as pl

import torch
from vllm import LLM, SamplingParams

torch.set_float32_matmul_precision("medium")


class StateEmbeddingModel:
    def __init__(self, config) -> None:
        super().__init__()
        self.model = LLM(**config.embed_params)
        self.tokenizer =self.model.get_tokenizer()
        self.max_len = config.max_embed_tokens

    # todo batch
    def encode(self, state):

        prompt_token_ids = self.tokenizer.encode('query: ' + state[0])#, return_tensors="pt")

        # Truncate prompt_token_ids
        prompt_token_ids = prompt_token_ids[:self.max_len]


        emb = self.model.encode(prompt_token_ids=prompt_token_ids, use_tqdm=False, )[
            0].outputs.data

        # emb = self.model.encode('query: ' + state[0], use_tqdm=False, )[
        #     0].outputs.data

        return [emb]


