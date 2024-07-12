"""Lightning module for the tactic generator."""

import re
from subprocess import CalledProcessError
from typing import Dict, Any
from typing import Optional, List

import torch
from lean_dojo.utils import execute
from loguru import logger
from torchmetrics import Metric
from torchmetrics.text import SacreBLEUScore
from transformers import T5ForConditionalGeneration, T5EncoderModel, AutoTokenizer

from experiments.end_to_end.common import remove_marks
from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

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

    def filter_tacs(self, tactics: List[str], num_filtered: int, goal, theorem) -> List[str]:
        state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal
        state = [t + theorem + '\n\n' + state for t in tactics]

        # todo chunk into batches for speedup

        tokenized_goals = (self.tokenizer(
                goal,
                padding=None,
                max_length=int(self.max_seq_len * 1.5),
                truncation=True,
                return_tensors="pt",))

        tokenized_tactics = []
        for goal in state:
            tokenized_goals.append(self.tokenizer(
                goal,
                padding=None,
                max_length=int(self.max_seq_len * 1.5),
                truncation=True,
                return_tensors="pt",))



        tokenized_tactics = self.tokenizer(
            tactics,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )


        lens = tokenized_tactics.attention_mask.sum(dim=1)


        encs = self.encoder.get_tac_encodings(tokenized_goals, mask, lens)
