"""Lightning module for the tactic generator."""

from typing import Dict, Any, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.text import SacreBLEUScore, ROUGEScore
from transformers import T5EncoderModel, T5ForConditionalGeneration
from transformers.utils import ModelOutput

from experiments.end_to_end.lightning_common import get_optimizers, load_checkpoint
from models.end_to_end.tactic_models.generator.model import TopkAccuracy
from loguru import logger

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score

torch.set_float32_matmul_precision("medium")

normalizer = lambda x: x


class BatchTacEmbed(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bleu = SacreBLEUScore()

        self.rogue = ROUGEScore(
            normalizer=lambda x: x,
            rouge_keys=("rouge1", "rouge2",
                        "rouge3", "rouge4",
                        "rouge5", "rouge6",
                        "rouge7", "rouge8",
                        "rouge9", "rougeL", "rougeLsum"),
        )

        if hasattr(config, 'load_ckpt') and config.load_ckpt:
            logger.info(f'Loading pretrained checkpoint..')
            ckpt = torch.load(config.ckpt_path)

            state_dict = {k[12:]: v for k, v in ckpt.items() if k.startswith('tac_encoder')}
            self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_encoder, state_dict=state_dict)

            self.score_network = torch.nn.Sequential(
                torch.nn.Linear(self.tac_encoder.config.d_model, self.tac_encoder.config.d_model // 2),
                torch.nn.LayerNorm(self.tac_encoder.config.d_model // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.tac_encoder.config.d_model // 2, 2),
            )

            state_dict = {k[14:]: v for k, v in ckpt.items() if k.startswith('score_network')}
            self.score_network.load_state_dict(state_dict)

            state_dict = {k[13:]: v for k, v in ckpt.items() if k.startswith('goal_encoder')}
            self.goal_encoder = T5EncoderModel.from_pretrained(config.goal_encoder, state_dict=state_dict)

            state_dict = {k[8:]: v for k, v in ckpt.items() if k.startswith('decoder')}
            self.decoder = T5ForConditionalGeneration.from_pretrained(config.decoder, state_dict=state_dict)
        else:
            self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_encoder)
            self.goal_encoder = T5EncoderModel.from_pretrained(config.goal_encoder)
            self.decoder = T5ForConditionalGeneration.from_pretrained(config.decoder)
            self.score_network = torch.nn.Sequential(
                torch.nn.Linear(self.tac_encoder.config.d_model, self.tac_encoder.config.d_model // 2),
                torch.nn.LayerNorm(self.tac_encoder.config.d_model // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.tac_encoder.config.d_model // 2, 2),
            )

        self.max_seq_len = config.max_length
        self.num_samples = config.num_samples
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps

        self.topk_accuracies = dict()
        for k in range(1, self.num_samples + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

        # add error and time predictors to the model, as an MLP with a single hidden layer
        # takes a single tactic vector and

        self.label_weights = config.label_weights

        self.error_weight = config.error_weight
        self.time_weight = config.time_weight

        self.ce_loss = CrossEntropyLoss(weight=torch.tensor(self.label_weights))
        self.bcm = BinaryConfusionMatrix()  # normalize='true')

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def get_goal_encoding(self, goal_ids, goal_mask):
        goal_enc = self.goal_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        lens = goal_mask.sum(dim=1)

        goal_enc_ = (goal_enc * goal_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        goal_enc_ = F.normalize(goal_enc_, dim=1).unsqueeze(1)
        return goal_enc_

    def get_full_encoding(self,
                          goal_ids: torch.Tensor,
                          goal_mask: torch.Tensor,
                          tactic_ids: torch.Tensor,
                          tactic_mask: torch.Tensor,
                          ):

        goal_enc = self.goal_encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        lens = goal_mask.sum(dim=1)

        goal_enc_ = (goal_enc * goal_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        goal_enc_ = F.normalize(goal_enc_, dim=1).unsqueeze(1)

        tac_embeds = self.tac_encoder.encoder.embed_tokens(tactic_ids)

        # set first embedding to be the pooled goal encoding
        tac_embeds_with_goal = torch.cat([goal_enc_, tac_embeds], dim=1)

        new_mask = torch.cat([torch.ones(tactic_mask.shape[0], 1).to(self.device), tactic_mask], dim=1)

        tac_enc = self.tac_encoder(inputs_embeds=tac_embeds_with_goal, attention_mask=new_mask,
                                   return_dict=True).last_hidden_state

        lens = new_mask.sum(dim=1)

        tac_enc = (tac_enc * new_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        tac_enc = F.normalize(tac_enc, dim=1).unsqueeze(1)

        # give full goal with tac/goal embed for better decoding
        full_enc = torch.cat([tac_enc, goal_enc], dim=1)
        # new_mask = torch.cat([torch.ones(goal_mask.shape[0], 1).to(self.device), goal_mask], dim=1)
        #
        # full_enc = self.goal_encoder(inputs_embeds=goal_embeds_with_tac, attention_mask=new_mask,
        #                              return_dict=True).last_hidden_state

        return full_enc.bfloat16(), tac_enc.squeeze(1).bfloat16()

    def forward(
            self,
            goal_ids: torch.Tensor,
            goal_mask: torch.Tensor,
            tactic_ids: torch.Tensor,
            tactic_mask: torch.Tensor,
            result_ids: torch.Tensor,
            error_targets: torch.Tensor,
            time_targets: torch.Tensor):
        full_enc, tac_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_ids, tactic_mask)

        dec_loss = self.decoder(
            encoder_outputs=(full_enc,),
            labels=result_ids,
        ).loss

        # batch_size x 2 (error, time)
        score_output = self.score_network(tac_enc)  # .squeeze(1)
        error_preds = torch.sigmoid(score_output[:, 0])
        time_preds = score_output[:, 1]

        error_preds = error_preds.unsqueeze(1)
        error_preds = torch.cat([1 - error_preds, error_preds], dim=1)

        error_loss = self.ce_loss(error_preds, error_targets)
        time_loss = F.mse_loss(time_preds, time_targets)

        return dec_loss, error_loss, time_loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        # error_targets = torch.tensor(
        #     [1. if batch['status'][i] == 'success' else 0. for i in range(len(batch['status']))],
        #     dtype=torch.bfloat16).to(self.device)

        error_targets = torch.tensor(
            [1 if batch['status'][i] == 'success' else 0 for i in range(len(batch['status']))],
            dtype=torch.long).to(self.device)

        dec_loss, error_loss, time_loss = self(
            batch["goal_ids"],
            batch["goal_mask"],
            batch["tactic_ids"],
            batch["tactic_mask"],
            batch["result_ids"],
            error_targets,
            batch["time_targets"],
        )

        self.log(
            "dec_loss_train",
            dec_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "time_loss_train",
            time_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        self.log(
            "error_loss_train",
            error_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        return dec_loss + self.error_weight * error_loss + self.time_weight * time_loss

    ##############
    # Validation #
    ##############

    def on_validation_epoch_start(self) -> None:
        if self.global_rank == 0:
            # using columns and data
            self.log_table = []

    def on_validation_epoch_end(self) -> None:
        if self.global_rank == 0:
            self.logger.log_table(key=f'predictions_{self.global_step}',
                                  columns=["goal", "tactic", "outcome", "time", "outcome_prediction",
                                           "error_prediction",
                                           "error_probs",
                                           "time_prediction"],
                                  data=self.log_table)

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        goal_ids = batch["goal_ids"]
        goal_mask = batch["goal_mask"]
        tactic_ids = batch["tactic_ids"]
        tactic_mask = batch["tactic_mask"]
        result_ids = batch["result_ids"]
        time_targets = batch["time_targets"]

        full_enc, tac_enc = self.get_full_encoding(goal_ids, goal_mask, tactic_ids, tactic_mask)

        dec_loss = self.decoder(
            encoder_outputs=(full_enc,),
            labels=result_ids,
        ).loss

        self.log(f"dec_loss_val", dec_loss, on_step=False, on_epoch=True, sync_dist=True)

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

        predictions = [
            output_text[i * self.num_samples: (i + 1) * self.num_samples]
            for i in range(batch_size)
        ]

        error_targets = torch.LongTensor(
            [1 if batch['status'][i] == 'success' else 0 for i in range(len(batch['status']))]).to(self.device)

        score_output = self.score_network(tac_enc)  # .squeeze(1)
        error_probs = torch.sigmoid(score_output[:, 0])

        time_preds = score_output[:, 1]
        time_loss = F.mse_loss(time_preds, time_targets)

        self.log(f'time_loss_val', time_loss, on_step=False, on_epoch=True, prog_bar=False)

        # get preds as those > 0.5
        error_preds = error_probs > 0.5
        # make 1 for true, 0 for false
        error_preds = error_preds.int()

        confusion = self.bcm(error_preds, error_targets)

        self.log(
            "false_negs",
            confusion[1][0],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=False,
            reduce_fx='sum'
        )

        self.log(
            "true_negs",
            confusion[0][0],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=False,
            reduce_fx='sum'
        )

        self.log(
            "false_pos",
            confusion[0][1],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=False,
            reduce_fx='sum'
        )

        self.log(
            "true_pos",
            confusion[1][1],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=False,
            reduce_fx='sum'
        )

        # check if the status is correct using error_preds
        error_acc = sum(
            [error_preds[i] == error_targets[i] for i in range(len(error_preds))]) / len(
            error_preds)

        # only consider the first prediction for decoder error accuracy
        dec_error_acc = sum(
            [predictions[i][0].split('\n')[0] == batch['status'][i] for i in range(batch_size)]) / batch_size

        self.log(f"error_acc", error_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"dec_error_acc", dec_error_acc, on_step=False, on_epoch=True, prog_bar=True)

        for k in range(1, self.num_samples + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(predictions, batch["result"])
            self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True, prog_bar=k == self.num_samples)

        assert len(output_text) == batch_size * self.num_samples, (
            len(output_text), batch_size, self.num_samples)

        bleu_targets = [[batch['result'][i]] for i in range(batch_size) for _ in range(self.num_samples)]

        nl = '\n\n'

        # self.log_dict(self.rogue(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)
        #
        # self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log_dict(rouge_score(output_text, bleu_targets, normalizer=normalizer), on_step=False, on_epoch=True,
                      prog_bar=False)

        # self.log('val_bleu', self.bleu(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_bleu', sacre_bleu_score(output_text, bleu_targets), on_step=False, on_epoch=True, prog_bar=False)

        self.log('avg_seq_len', sum([len(o) for o in output_text]) / len(output_text), on_step=False, on_epoch=True,
                 prog_bar=False)

        # log table to wandb for rank 0 only
        if self.global_rank == 0:
            data = [[batch['goal'][i], batch['tactic'][i], batch['result'][i], batch['time_targets'][i],
                     nl.join(predictions[i]), 'success' if error_preds[i] == 1 else 'failure', error_probs[i],
                     time_preds[i]]
                    for i in range(batch_size)]
            self.log_table.extend(data)
