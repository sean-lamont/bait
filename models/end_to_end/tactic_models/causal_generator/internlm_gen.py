"""Lightning module for the tactic generator."""

import re
from subprocess import CalledProcessError
from typing import Dict, Any
from typing import Optional, List

import torch
from lean_dojo.utils import execute
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from torchmetrics import Metric
from torchmetrics.text import SacreBLEUScore

from experiments.end_to_end.common import remove_marks, zip_strict, format_augmented_state
from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

torch.set_float32_matmul_precision("medium")


class InternLMGenerator(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def batch_generate(self, state, retriever_args, num_samples):
        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # todo max_length or return tokens
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            num_return_sequences=num_samples,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=0.7
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        transitions = self.generator.compute_transition_scores(output.sequences, output.scores,
                                                               normalize_logits=True)

        output_text = []
        output_score = []

        for j in range(len(raw_output_text)):
            t = raw_output_text[j]
            t = t.split('TACTIC: ')[-1]
            if t not in output_text:
                output_text.append(t)
                score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
                output_score.append(score)

        tactics_with_scores = list(zip_strict(output_text, output_score))

        return [tactics_with_scores], [state]

    # Following the paper, only samples for now, with fixed temperature of 0.5

    #
    #     # return state_with_prompt as well to store retrieved state_with_prompt
    #     if self.gen_config.strategy == 'sample':
    #         return self.sample_gen(state, state_ids, state_mask, num_samples), state
    #     elif self.gen_config.strategy == 'beam':
    #         return self.beamsearch_gen(state, state_ids, state_mask, num_samples), state
    #     else:
    #         raise NotImplementedError
    #
    # def sample_gen(self, state, state_ids, state_mask, num_samples):
    #     # score for nucleus sampling
    #     tactics_with_scores = []
    #
    #     output_text = []
    #     output_score = []
    #     gen_step = 0
    #
    #     gen_idx = 0
    #     # keep sampling until num_samples unique samples are generated, with at most 10 loops
    #     while len(output_text) < num_samples and gen_idx < 10:
    #         gen_idx += 1
    #         output = self.generator.generate(
    #             input_ids=state_ids,
    #             attention_mask=state_mask,
    #             max_length=self.max_seq_len,
    #             do_sample=True,
    #             num_return_sequences=2,
    #             output_scores=True,
    #             return_dict_in_generate=True,
    #             top_p=0.9,
    #         )
    #
    #         transitions = self.generator.compute_transition_scores(output.sequences, output.scores,
    #                                                                normalize_logits=True)
    #         # Return the output.
    #         raw_output_text = self.tokenizer.batch_decode(
    #             output.sequences, skip_special_tokens=True
    #         )
    #
    #         for j in range(len(raw_output_text)):
    #             t = raw_output_text[j]
    #             t = t.split('TACTIC:')[-1]
    #             if t not in output_text:
    #                 output_text.append(t)
    #                 score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
    #                 output_score.append(score)
    #             if len(output_text) >= num_samples:
    #                 break
    #
    #         gen_step += 1
    #
    #     tactics_with_scores.append(list(zip_strict(output_text, output_score))[:num_samples])
    #
    #     return tactics_with_scores
    #
    # def beamsearch_gen(self, state, state_ids, state_mask, num_samples):
    #     # Generate tactic candidates using beam search.
    #     output = self.generator.generate(
    #         input_ids=state_ids,
    #         attention_mask=state_mask,
    #         max_length=self.max_seq_len,
    #         num_beams=num_samples,
    #         length_penalty=self.gen_config.length_penalty,
    #         do_sample=False,
    #         num_return_sequences=num_samples,
    #         # early_stopping=False,
    #         early_stopping=True,
    #         output_scores=True,
    #         return_dict_in_generate=True,
    #     )
    #
    #     # Return the output.
    #     raw_output_text = self.tokenizer.batch_decode(
    #         output.sequences, skip_special_tokens=True
    #     )
    #
    #     raw_scores = output.sequences_scores.tolist()
    #     tactics_with_scores = []
    #
    #     for i in range(len(state)):
    #         output_text = []
    #         output_score = []
    #
    #         for j in range(i * num_samples, (i + 1) * num_samples):
    #             t = raw_output_text[j]
    #             t = t.split('[ANSWER]')[-1]
    #             if t not in output_text:
    #                 output_text.append(t)
    #                 output_score.append(raw_scores[j])
    #
    #         tactics_with_scores.append(list(zip_strict(output_text, output_score)))
    #
    #     return tactics_with_scores
