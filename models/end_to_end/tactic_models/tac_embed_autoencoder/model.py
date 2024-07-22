"""Lightning module for the tactic generator."""

from typing import Dict, Any
import lightning.pytorch as pl

import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer, T5ForConditionalGeneration
import torch
from torchmetrics.text import SacreBLEUScore
from loguru import logger
from transformers.utils import ModelOutput

from experiments.end_to_end.lightning_common import get_optimizers, load_checkpoint, cpu_checkpointing_enabled

torch.set_float32_matmul_precision("medium")


class AutoEncoderTacModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bleu = SacreBLEUScore()

        self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_model)

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
                          tactic_ids: torch.Tensor,
                          tactic_mask: torch.Tensor,
                          ):

        tac_enc = self._encode(self.tac_encoder, tactic_ids, tactic_mask).unsqueeze(1)

        return tac_enc

    def forward(
            self,
            tactic_ids: torch.Tensor,
            tactic_mask: torch.Tensor,
            result_ids: torch.Tensor,
            result_mask: torch.Tensor,
    ) -> torch.Tensor:

        full_enc = self.get_full_encoding(tactic_ids, tactic_mask)

        return self.decoder(
            encoder_outputs=(full_enc,),
            labels=result_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
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

    def on_validation_epoch_start(self) -> None:
        # using columns and data
        self.log_table = []

    def on_validation_epoch_end(self) -> None:
        self.logger.log_table(key=f'auto_enc_val_predictions_{self.global_step}',
                              columns=["tactic", "prediction"],
                              data=self.log_table)

    def validation_step(self, batch: Dict[str, Any], _) -> None:

        tactic_ids = batch["tactic_ids"]
        tactic_mask = batch["tactic_mask"]
        result_ids = batch["result_ids"]

        full_enc = self.get_full_encoding(tactic_ids, tactic_mask)

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

        batch_size = tactic_ids.size(0)

        assert len(output_text) == batch_size * self.num_samples, (
            len(output_text), batch_size, self.num_samples)

        # for us, we only have one target (reference) so targets will be a list of lists,
        # with targets[i * num_samples: (i+1) * num_samples] being the target for the corresponding sample
        bleu_targets = [
            batch['tactic']
            for _ in range(batch_size)
            for _ in range(self.num_samples)
        ]

        nl = '\n'

        self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log('avg_seq_len', sum([len(o) for o in output_text]) / len(output_text), on_step=False, on_epoch=True,
                 prog_bar=False)

        data = [[batch['tactic'][i],
                 nl.join(output_text[i * self.num_samples: (i + 1) * self.num_samples])]
                for i in range(batch_size)]

        self.log_table.extend(data)
