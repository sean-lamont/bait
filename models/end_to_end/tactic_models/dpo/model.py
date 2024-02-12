"""Lightning module for the tactic generator."""

import torch
import torch.nn.functional as F
from loguru import logger

from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

torch.set_float32_matmul_precision("medium")


class DPOTrainModule(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.beta = config.beta

        self.save_hyperparameters()

    def dpo_loss(self, pi_yw_logps, pi_yl_logps, ref_yw_logps, ref_yl_logps):
        """
        pi_yw_logps: policy logprobs for winners
        ref_yw_logps: reference model logprobs for winners
        pi_yl_logps: policy logprobs for losers
        ref_yl_logps: reference model logprobs for losers
        beta: temperature controlling strength of KL penalty
        """

        # pi_logps = torch.cat([pi_yw_logps, pi_yl_logps], dim=0)
        # ref_logps = torch.cat([ref_yw_logps, ref_yl_logps], dim=0)
        # rewards = beta * (pi_logps - ref_logps).detach()

        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        return losses  # , rewards

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            winner_ids: torch.Tensor,
            loser_ids: torch.Tensor,
            winner_ref_probs: torch.Tensor,
            loser_ref_probs: torch.Tensor,
    ) -> torch.Tensor:
        winner_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=winner_ids,
        ).logits

        loser_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=loser_ids
        ).logits

        winner_ids[winner_ids == -100] = 0
        loser_ids[loser_ids == -100] = 0

        # construct probability for whole sequence (from testing, slightly off reported beam search scores by ~10^-2)
        normalised = torch.log_softmax(winner_pi_logits, -1)
        outs = torch.gather(normalised, -1, winner_ids.unsqueeze(-1)).squeeze(-1)
        outs[winner_ids == self.tokenizer.pad_token_id] = 0
        winner_pi_probs = torch.sum(outs, dim=1)

        normalised = torch.log_softmax(loser_pi_logits, -1)
        outs = torch.gather(normalised, -1, loser_ids.unsqueeze(-1)).squeeze(-1)
        outs[loser_ids == self.tokenizer.pad_token_id] = 0
        loser_pi_probs = torch.sum(outs, dim=1)

        loss = self.dpo_loss(winner_pi_probs, loser_pi_probs, winner_ref_probs, loser_ref_probs)

        return torch.sum(loss, dim=0)

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["winner_ids"],
            batch["loser_ids"],
            batch["winner_ref_probs"],
            batch["loser_ref_probs"],
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

    def validation_step(self, batch, batch_idx):
        state_ids = batch['state_ids']
        state_mask = batch['state_mask']
        winner_ids = batch['winner_ids']
        loser_ids = batch['loser_ids']

        winner_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=winner_ids,
        ).logits

        loser_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=loser_ids
        ).logits

        winner_ids[winner_ids == -100] = 0
        loser_ids[loser_ids == -100] = 0

        # construct probability for whole sequence (from testing, slightly off reported beam search scores by ~10^-2)
        normalised = torch.log_softmax(winner_pi_logits, -1)
        outs = torch.gather(normalised, -1, winner_ids.unsqueeze(-1)).squeeze(-1)
        outs[winner_ids == self.tokenizer.pad_token_id] = 0
        winner_pi_probs = torch.sum(outs, dim=1)

        normalised = torch.log_softmax(loser_pi_logits, -1)
        outs = torch.gather(normalised, -1, loser_ids.unsqueeze(-1)).squeeze(-1)
        outs[loser_ids == self.tokenizer.pad_token_id] = 0
        loser_pi_probs = torch.sum(outs, dim=1)

        # when difference is > 0, winner_probs is greater
        preferred = sum((winner_pi_probs - loser_pi_probs) > 0) / winner_pi_probs.shape[0]

        logger.debug(
            f'{torch.cat([winner_pi_probs, batch["winner_ref_probs"], loser_pi_probs, batch["loser_ref_probs"]], dim=0)}')

        self.log(
            "win_rate",
            preferred,
            on_epoch=True,
            sync_dist=True,
            batch_size=winner_pi_probs.shape[0],
            prog_bar=True
        )
