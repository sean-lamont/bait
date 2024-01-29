"""Lightning module for the tactic generator."""

from typing import Dict, Any, Optional, List

import torch
from loguru import logger
from torchmetrics import Metric

from experiments.end_to_end.common import remove_marks
from models.end_to_end.tactic_models.dpo.model import GenTacModel

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

        # self.save_hyperparameters()

        self.num_beams = config.num_beams

        ret_ckpt_path = config.ret_ckpt_path

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            # self.retriever = PremiseRetriever.load(
            #     ret_ckpt_path, self.device, freeze=True
            # )

        self.topk_accuracies = dict()
        for k in range(1, self.num_beams + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

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

        loss = self(state_ids, state_mask, tactic_ids)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)

        output_text = []
        for s in state:
            # Generate topk tactic candidates
            output = self.generate(s, retriever_args=None, num_samples=self.num_beams)

            output = [o[0] for o in output]

            # fill in with blanks if full beams are not generated
            for _ in range(len(output), self.num_beams):
                output.append('')

            output_text.extend(output)

        batch_size = state_ids.size(0)

        assert len(output_text) == batch_size * self.num_beams, (len(output_text), batch_size, self.num_beams)

        tactics_pred = [
            output_text[i * self.num_beams: (i + 1) * self.num_beams]
            for i in range(batch_size)
        ]

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            if k == 16:
                self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True, prog_bar=True)