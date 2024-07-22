"""Lightning module for the tactic generator."""

from typing import Dict, Any
import lightning.pytorch as pl

import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer, T5ForConditionalGeneration
import torch
from torchmetrics.text import SacreBLEUScore
from loguru import logger
from transformers.utils import ModelOutput

from experiments.end_to_end.lightning_common import get_optimizers, load_checkpoint

torch.set_float32_matmul_precision("medium")

class TransitionModelLarge(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bleu = SacreBLEUScore()

        # self.goal_tokenizer = AutoTokenizer.from_pretrained(config.encoder)
        self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_encoder)

        # self.goal_tokenizer = AutoTokenizer.from_pretrained(config.goal_model)
        self.goal_encoder = T5EncoderModel.from_pretrained(config.goal_encoder)

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

    def get_tac_encoding(self, goal_ids, goal_mask, tactic_lens):
        # encode all tokens with tactic included
        combined_enc = self.tac_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        # get the tactic embeddings and mean pool them using the provided lengths
        tac_enc = []

        for i in range(combined_enc.shape[0]):
            enc = combined_enc[i, :tactic_lens[i]]
            enc = enc.sum(dim=0) / tactic_lens[i]
            enc = F.normalize(enc, dim=0)
            tac_enc.append(enc)

        tac_enc = torch.stack(tac_enc, dim=0).unsqueeze(1)
        return tac_enc

    # bottleneck information to single tactic vec
    def get_full_encoding(self,
                          goal_ids: torch.Tensor,
                          goal_mask: torch.Tensor,
                          tactic_lens: torch.Tensor,
                          ):
        # encode all tokens with tactic included
        combined_enc = self.tac_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        # get the tactic embeddings and mean pool them using the provided lengths
        tac_enc = []
        new_ids = goal_ids.clone()

        for i in range(combined_enc.shape[0]):
            enc = combined_enc[i, :tactic_lens[i]]
            enc = enc.sum(dim=0) / tactic_lens[i]
            enc = F.normalize(enc, dim=0)
            tac_enc.append(enc)
            # zero out ids for tactics in combined (tac, goal) from goal_ids, so there is no information for the
            # goal encoder
            new_ids[i, :tactic_lens[i]] = 0

        tac_enc = torch.stack(tac_enc, dim=0).unsqueeze(1)

        goal_embeds = self.goal_encoder.encoder.embed_tokens(new_ids)

        # set first embedding to be the tactic encoding
        goal_embeds_with_tac = torch.cat([tac_enc, goal_embeds], dim=1)

        new_mask = torch.cat([torch.ones(goal_mask.shape[0], 1).to(self.device), goal_mask], dim=1)

        full_enc = self.goal_encoder(inputs_embeds=goal_embeds_with_tac, attention_mask=new_mask,
                                     return_dict=True).last_hidden_state

        return full_enc


    # encoding which allows goal token embeddings to have attended to tactic
    # (more accurate, but less information forced into tactic)

    # def get_full_encoding(self,
    #                       goal_ids: torch.Tensor,
    #                       goal_mask: torch.Tensor,
    #                       tactic_lens: torch.Tensor,
    #                       ):
    #
    #     # encode all tokens with tactic included
    #     goal_enc = self.encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state
    #
    #     # get the tactic embeddings and mean pool them using the provided lengths
    #     tac_enc = []
    #
    #     for i in range(goal_enc.shape[0]):
    #         enc = goal_enc[i, :tactic_lens[i]]
    #         enc = enc.sum(dim=0) / tactic_lens[i]
    #         enc = F.normalize(enc, dim=0)
    #         tac_enc.append(enc)
    #         # todo better way than zeroing out original tactic tokens?
    #         goal_enc[i, :tactic_lens[i]] = 0
    #
    #     tac_enc = torch.stack(tac_enc, dim=0).unsqueeze(1)
    #
    #     full_enc = torch.cat([goal_enc, tac_enc], dim=1)
    #
    #     return full_enc

    def forward(
            self,
            goal_ids: torch.Tensor,
            goal_mask: torch.Tensor,
            tactic_lens: torch.Tensor,
            result_ids: torch.Tensor,
            result_mask: torch.Tensor,
    ) -> torch.Tensor:
        full_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_lens)

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
            batch["tactic_lens"],
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
        self.logger.log_table(key=f'large_val_predictions_{self.global_step}',
                              columns=["goal", "tactic", "outcome", "prediction"],
                              data=self.log_table)

    def validation_step(self, batch: Dict[str, Any], _) -> None:

        goal_ids = batch["goal_ids"]
        goal_mask = batch["goal_mask"]
        tactic_lens = batch["tactic_lens"]
        result_ids = batch["result_ids"]

        full_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_lens)

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

        nl = '\n\n'

        # logger.info(f'Goal Before:\n {batch["goal"][0]}\n\n Goal After:\n  {batch["result"][0]} \n\n Predicted: \n{nl.join([o for o in output_text])}\n\n\n,')

        self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log('avg_seq_len', sum([len(o) for o in output_text]) / len(output_text), on_step=False, on_epoch=True,
                 prog_bar=False)

        data = [[batch['goal'][i], batch['tactic'][i], batch['result'][i],
                 nl.join(output_text[i * self.num_samples: (i + 1) * self.num_samples])]
                for i in range(batch_size)]

        self.log_table.extend(data)
