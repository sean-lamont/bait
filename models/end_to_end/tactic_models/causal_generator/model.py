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

from experiments.end_to_end.common import remove_marks, zip_strict, format_augmented_state
from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

torch.set_float32_matmul_precision("medium")


# todo saving model checkpoint as LoRA weights
# https://github.com/Lightning-AI/pytorch-lightning/issues/19228


class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class RetrievalAugmentedGenerator(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        # number of candidate tactics generated per goal
        self.num_val_samples = config.num_val_samples if hasattr(config, 'num_val_samples') else 0

        self.save_hyperparameters()

        self.topk_accuracies = dict()
        for k in range(1, self.num_val_samples + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

        self.bleu = SacreBLEUScore()

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        return loss

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        state = batch["state"]
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        retriever_args = batch["retriever_args"] if "retriever_args" in batch else None

        loss = self(state_ids, state_mask, tactic_ids)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)

        output_text = []
        for s in state:
            # Generate topk tactic candidates
            output, _ = self.generate(s, retriever_args=retriever_args, num_samples=self.num_val_samples)

            output = [o[0] for o in output]

            # fill in with blanks if full beams are not generated
            for _ in range(len(output), self.num_val_samples):
                output.append('')

            output_text.extend(output)

        batch_size = state_ids.size(0)

        assert len(output_text) == batch_size * self.num_val_samples, (
            len(output_text), batch_size, self.num_val_samples)

        tactics_pred = [
            output_text[i * self.num_val_samples: (i + 1) * self.num_val_samples]
            for i in range(batch_size)
        ]

        # print (tactics_pred)

        # Log the topk accuracies.
        for k in range(1, self.num_val_samples + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True, prog_bar=False)

        # for us, we only have one target (reference) so targets will be a list of lists,
        # with targets[i * num_val_samples: (i+1) * num_val_samples] being the target for the corresponding sample
        bleu_targets = [
            [batch['tactic'][i]]
            for i in range(batch_size)
            for _ in range(self.num_val_samples)
        ]

        self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log('avg_seq_len', sum([len(o) for o in output_text]) / len(output_text), on_step=False, on_epoch=True,
                 prog_bar=False)

    def run_eval(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last_eval.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        cmd = f"python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/train/gen_seq2seq/eval num_theorems={self.eval_config.eval_num_theorems}" \
              f" shuffle={self.eval_config.shuffle} env_timeout={self.eval_config.timeout} tac_model.ckpt_path={ckpt_path} log_level='ERROR' tac_model.model='reprover'" \
              f" exp_config.name=eval_epoch_{self.trainer.current_epoch} exp_config.experiment=seq2seq_eval" \
              f" num_iterations=1"

        logger.info(f'Running evaluation with {cmd}')

        try:
            _, err = execute(cmd, capture_output=True)
        except CalledProcessError as ex:
            logger.error(ex)
            logger.error("Failed to evaluate.")
            return

        m = re.search(r"Pass@1: (\S+)", err)
        assert m is not None, err
        acc = float(m.group(1))
        self.log("Pass@1_val", acc, prog_bar=True)
        logger.info(f"Pass@1: {acc}")

    # todo: update generation to:
    # - Parse out response (take text after [ANSWER] and ensure it's in the correct format)
    # -

    def batch_generate(self, state, retriever_args, num_samples):
        prompt = ('You are an expert in Lean 3 theorem proving.'
                  ' Suggest a tactic to solve the following goal.'
                  ' Any premises in the tactic should be included in the following format: <a>premise<\\a>.'
                  'Return your answer in the following format: [ANSWER]your_tactic\n\n')

        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                retriever_args,
                self.eval_num_retrieved,
            )

            # todo get exact prompt length after tokenization
            updated_len = self.max_seq_len - len(prompt)

            state = [
                format_augmented_state(s, premises, updated_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        state = [prompt + s for s in state]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # return state as well to store retrieved state
        if self.gen_config.strategy == 'sample':
            return self.sample_gen(state, state_ids, state_mask, num_samples), state
        elif self.gen_config.strategy == 'beam':
            return self.beamsearch_gen(state, state_ids, state_mask, num_samples), state
        else:
            raise NotImplementedError

    def sample_gen(self, state, state_ids, state_mask, num_samples):
        # score for nucleus sampling
        tactics_with_scores = []

        output_text = []
        output_score = []
        gen_step = 0

        gen_idx = 0
        # keep sampling until num_samples unique samples are generated, with at most 10 loops
        while len(output_text) < num_samples and gen_idx < 10:
            gen_idx += 1
            output = self.generator.generate(
                input_ids=state_ids,
                attention_mask=state_mask,
                max_length=self.max_seq_len,
                do_sample=True,
                num_return_sequences=num_samples,
                output_scores=True,
                return_dict_in_generate=True,
            )

            transitions = self.generator.compute_transition_scores(output.sequences, output.scores,
                                                                   normalize_logits=True)
            # Return the output.
            raw_output_text = self.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )

            for j in range(num_samples * 2):
                t = raw_output_text[j]
                if t not in output_text:
                    output_text.append(t)
                    score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
                    output_score.append(score)
                if len(output_text) >= num_samples:
                    break

            gen_step += 1

        tactics_with_scores.append(list(zip_strict(output_text, output_score))[:num_samples])

        return tactics_with_scores

    def beamsearch_gen(self, state, state_ids, state_mask, num_samples):
        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.gen_config.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            # early_stopping=False,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = raw_output_text[j]
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores
