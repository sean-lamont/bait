"""Lightning module for the tactic generator."""
import pickle
from typing import List
from typing import Tuple

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP
from transformers import T5EncoderModel, AutoTokenizer
import torch.nn.functional as F
from loguru import logger

torch.set_float32_matmul_precision("medium")


class DiversityModel(torch.nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.encoder, self.tokenizer = self.load_encoder(config)
        self.max_seq_len = config.max_seq_len

    def load_encoder(self, config):
        if config.ckpt_dir:
            ckpt = torch.load(config.ckpt_dir)
            state_dict = {k[12:]: v for k, v in ckpt.items() if k.startswith('tac_encoder')}

            tac_encoder = T5EncoderModel.from_pretrained(config.model,
                                                         state_dict=state_dict).to(self.device)
        else:
            tac_encoder = T5EncoderModel.from_pretrained(config.model).to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(config.model)

        return tac_encoder, tokenizer

    def get_tac_encoding(self, goal_ids, goal_mask, tactic_lens):
        # encode all tokens with tactic included
        combined_enc = self.encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        # get the tactic embeddings and mean pool them using the provided lengths
        tac_enc = []

        for i in range(combined_enc.shape[0]):
            enc = combined_enc[i, :tactic_lens[i]]
            enc = enc.sum(dim=0) / tactic_lens[i]
            enc = F.normalize(enc, dim=0)
            tac_enc.append(enc)

        tac_enc = torch.stack(tac_enc, dim=0).unsqueeze(1)
        return tac_enc

    def filter_tacs(self, tactics: List[Tuple[str, float]], num_filtered: int, state, theorem, temperature=1.,
                    scale=1e5):
        with torch.no_grad():
            # state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal

            encs = []

            logprobs = [t[1] / temperature for t in tactics]

            # get softmax over logprobs
            probs = torch.softmax(torch.tensor(logprobs), dim=0) * scale

            # chunking gives slight speedup, but high memory cost
            chunk_size = 1
            for ind in range(0, len(tactics), chunk_size):
                t = [t[0] for t in tactics[ind:ind + chunk_size]]

                goals = [t_ + theorem + '\n\n' + state for t_ in t]

                tokenized_goals = self.tokenizer(
                    goals,
                    padding="longest",
                    max_length=int(self.max_seq_len * 1.5),
                    truncation=True,
                    return_tensors="pt", )

                tokenized_tactics = self.tokenizer(
                    t,
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )

                lens = tokenized_tactics.attention_mask.sum(dim=1)

                enc = self.get_tac_encoding(tokenized_goals.input_ids.to(self.device),
                                            tokenized_goals.attention_mask.to(self.device), lens.to(self.device))

                enc = enc.squeeze(1)

                # scale enc by normalised tactic logprobs
                encs.append(enc)

            vec_matrix = torch.cat(encs, dim=0)
            vec_matrix = torch.mul(vec_matrix, probs.unsqueeze(1).to(self.device)).cpu().numpy()

            vec_matrix = vec_matrix @ vec_matrix.T


            # todo right now, must modify the DPP library code from np.float to float for version compatibility
            DPP = FiniteDPP('likelihood', **{'L': vec_matrix})

            # rng = np.random.RandomState(1)
            try:
                # DPP.sample_exact_k_dpp(size=num_filtered, mode='KuTa12')#, rng
                DPP.sample_exact()
            except Exception as e:
                logger.error(f"Error sampling from DPP: {e}, returning top {str(num_filtered)} tactics, out of {str(len(tactics))}")
                return [[i for i in range(num_filtered)]]

        return DPP.list_of_samples
