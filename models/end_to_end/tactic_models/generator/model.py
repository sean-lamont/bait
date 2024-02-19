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

from experiments.end_to_end.common import remove_marks
from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

torch.set_float32_matmul_precision("medium")


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


# todo retriever
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
    # Training
    #
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
            output = self.generate(s, retriever_args=retriever_args, num_samples=self.num_val_samples)

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
