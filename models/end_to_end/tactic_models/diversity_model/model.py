"""Lightning module for the tactic generator."""

from typing import List
from typing import Tuple
import numpy as np
from dppy.finite_dpps import FiniteDPP

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import T5EncoderModel, AutoTokenizer

torch.set_float32_matmul_precision("medium")


class DiversityModel(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder, self.tokenizer = self.load_encoder(config.enc_config)

    def load_encoder(self):
        if self.config.ckpt_dir:
            ckpt = torch.load(self.config.ckpt_dir)
            state_dict = {k[12:]: v for k, v in ckpt.items() if k.startswith('tac_encoder')}

            tac_encoder = T5EncoderModel.from_pretrained(self.config.model,
                                                         state_dict=state_dict).cuda()
        else:
            tac_encoder = T5EncoderModel.from_pretrained(self.config.model).cuda()

        tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        return tac_encoder, tokenizer

    def filter_tacs(self, tactics: List[Tuple[str, float]], num_filtered: int, goal, theorem) -> List[str]:
        state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal

        encs = []

        logprobs = [t[1] for t in tactics]

        # get softmax over logprobs
        probs = torch.softmax(torch.tensor(logprobs), dim=0)

        # todo chunk into batches enc speedup
        for t in tactics:
            goal = [t[0] + theorem + '\n\n' + state]

            tokenized_goals = self.tokenizer(
                goal,
                padding="longest",
                max_length=int(self.max_seq_len * 1.5),
                truncation=True,
                return_tensors="pt", )

            tokenized_tactics = self.tokenizer(
                tactics,
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )

            lens = tokenized_tactics.attention_mask.sum(dim=1)

            enc = self.encoder.get_tac_encodings(tokenized_goals.input_ids, tokenized_goals.attention_mask, lens)

            # scale enc by normalised tactic logprob
            enc = enc * probs[tactics.index(t)]
            encs.append(enc)

        vec_matrix = torch.stack(encs).numpy()

        rng = np.random.RandomState(1)

        DPP = FiniteDPP('likelihood', **{'L': vec_matrix})

        DPP.sample_exact_k_dpp(size=num_filtered, random_state=rng)

        return [t[0] for t in tactics][DPP.list_of_samples[0]]
