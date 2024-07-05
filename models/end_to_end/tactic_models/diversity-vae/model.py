"""Lightning module for the tactic generator."""

from typing import Dict, Any
import lightning.pytorch as pl

import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer, T5ForConditionalGeneration
import torch
from torchmetrics.text import SacreBLEUScore
from loguru import logger
from transformers.utils import ModelOutput

from experiments.end_to_end.common import cpu_checkpointing_enabled, load_checkpoint, get_optimizers

torch.set_float32_matmul_precision("medium")


class TransitionModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bleu = SacreBLEUScore()

        # self.tac_tokenizer = AutoTokenizer.from_pretrained(config.tac_model)
        self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_model)

        # self.goal_tokenizer = AutoTokenizer.from_pretrained(config.goal_model)
        self.goal_encoder = T5EncoderModel.from_pretrained(config.goal_model)

        # self.decoder_tokenizer = AutoTokenizer.from_pretrained(config.decoder)
        self.decoder = T5ForConditionalGeneration.from_pretrained(config.decoder)

        self.max_seq_len = config.max_length
        self.num_samples = config.num_samples
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    # def on_fit_start(self) -> None:
    #     if self.logger is not None and self.global_rank == 0:
    #         self.logger.log_hyperparams(self.hparams)
    #         assert self.trainer is not None
    #         logger.info(f"Logging to {self.trainer.log_dir}")

    def _encode(
            self, encoder, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def get_full_encoding(self,
                          goal_ids: torch.Tensor,
                          goal_mask: torch.Tensor,
                          tactic_ids: torch.Tensor,
                          tactic_mask: torch.Tensor,
                          ):

        tac_enc = self._encode(self.tac_encoder, tactic_ids, tactic_mask).unsqueeze(1)
        goal_enc = self.goal_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state
        full_enc = torch.cat([goal_enc, tac_enc], dim=1)

        # omit tactic encoding to test
        # tac_enc = self._encode(self.tac_encoder, tactic_ids, tactic_mask).unsqueeze(1)
        # goal_enc = self.goal_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state
        # full_enc = goal_enc
        # full_enc = torch.cat([goal_enc, tac_enc], dim=1)

        # print(tac_enc, goal_enc, full_enc, tac_enc.shape, goal_enc.shape, full_enc.shape)

        return full_enc

    def forward(
            self,
            goal_ids: torch.Tensor,
            goal_mask: torch.Tensor,
            tactic_ids: torch.Tensor,
            tactic_mask: torch.Tensor,
            result_ids: torch.Tensor,
            result_mask: torch.Tensor,
    ) -> torch.Tensor:

        full_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_ids, tactic_mask)

        return self.decoder(
            encoder_outputs=(full_enc,),
            labels=result_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["goal_ids"],
            batch["goal_mask"],
            batch["tactic_ids"],
            batch["tactic_mask"],
            batch["result_ids"],
            batch["result_mask"],
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
        goal_ids = batch["goal_ids"]
        goal_mask = batch["goal_mask"]
        tactic_ids = batch["tactic_ids"]
        tactic_mask = batch["tactic_mask"]
        result_ids = batch["result_ids"]

        full_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_ids, tactic_mask)

        loss = self.decoder(
            encoder_outputs=(full_enc,),
            labels=result_ids,
        ).loss

        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)

        enc_outs = ModelOutput(last_hidden_state=full_enc)

        output = self.decoder.generate(encoder_outputs=enc_outs,
                                       max_length=self.max_seq_len,
                                       num_beams=self.num_samples,
                                       do_sample=False,
                                       num_return_sequences=self.num_samples,
                                       early_stopping=True,
                                       output_scores=True,
                                       return_dict_in_generate=True,
                                       )

        # Return the output.
        output_text = self.trainer.datamodule.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        batch_size = goal_ids.size(0)

        assert len(output_text) == batch_size * self.num_samples, (
            len(output_text), batch_size, self.num_samples)

        # for us, we only have one target (reference) so targets will be a list of lists,
        # with targets[i * num_samples: (i+1) * num_samples] being the target for the corresponding sample
        bleu_targets = [
            [batch['result'][i]]
            for i in range(batch_size)
            for _ in range(self.num_samples)
        ]

        # nl= '\n'
        # logger.info(f'Goal Before:\n {batch["goal"][0]}\n\n Goal After:\n  {batch["result"][0]} \n\n Predicted: \n{nl.join([o for o in output_text])}\n\n\n,')

        self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log('avg_seq_len', sum([len(o) for o in output_text]) / len(output_text), on_step=False, on_epoch=True,
                 prog_bar=False)
