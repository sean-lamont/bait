"""Lightning module for the tactic generator."""
import copy
import traceback
from typing import Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.text import SacreBLEUScore, ROUGEScore
from transformers import T5EncoderModel, T5ForConditionalGeneration, AutoModel, BitsAndBytesConfig, AutoTokenizer
from transformers.utils import ModelOutput

from experiments.end_to_end.lightning_common import get_optimizers, load_checkpoint
from models.end_to_end.tactic_models.generator.model import TopkAccuracy
from loguru import logger

torch.set_float32_matmul_precision("medium")


# todo: InternLM transition model. Should try to batch all tactics with goal for efficiency.
# todo Then pool each of the tactics to get given embedding, followed by concat with original goal,
# todo then state prediction + prediction of original score

class InternLMTransitionModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.bleu = SacreBLEUScore()

        self.emb_dim = config.emb_dim

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
            # ckpt = torch.load(config.ckpt_path)
            #
            # state_dict = {k[12:]: v for k, v in ckpt.items() if k.startswith('tac_encoder')}
            # self.tac_encoder = T5EncoderModel.from_pretrained(config.tac_encoder, state_dict=state_dict)
            #
            # self.score_network = torch.nn.Sequential(
            #     torch.nn.Linear(self.tac_encoder.config.d_model, self.tac_encoder.config.d_model // 2),
            #     torch.nn.LayerNorm(self.tac_encoder.config.d_model // 2),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(self.tac_encoder.config.d_model // 2, 2),
            # )
            #
            # state_dict = {k[14:]: v for k, v in ckpt.items() if k.startswith('score_network')}
            # self.score_network.load_state_dict(state_dict)
            #
            # state_dict = {k[13:]: v for k, v in ckpt.items() if k.startswith('goal_encoder')}
            # self.goal_encoder = T5EncoderModel.from_pretrained(config.goal_encoder, state_dict=state_dict)
            #
            # state_dict = {k[8:]: v for k, v in ckpt.items() if k.startswith('decoder')}
            # self.decoder = T5ForConditionalGeneration.from_pretrained(config.decoder, state_dict=state_dict)
        else:

            # quant_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_quant_storage = torch.bfloat16,
            # )

            # quant_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_threshold=6.0
            # )

            self.enc_model = AutoModel.from_pretrained(
                "internlm/internlm2_5-step-prover-critic",
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # quantization_config=quant_config,
            )

            # dec_config = copy.deepcopy(self.enc_model.config)
            #
            # dec_config.architectures = ['InternLM2ForCausalLM']
            #
            # dec_config.auto_map = {"AutoConfig": "configuration_internlm2.InternLM2Config",
            #                        "AutoModel": "modeling_internlm2.InternLM2ForCausalLM"}

            # self.enc_model = AutoModel.from_pretrained(
            #     "internlm/internlm2_5-step-prover-critic",
            #     device_map="cuda",
            #     torch_dtype=torch.float16,
            #     trust_remote_code=True,
            #     quantization_config=quant_config,
            # )

            lora_config = LoraConfig(
                target_modules=[
                    "wqkv",
                    "wo",
                    "gate_up_proj",
                    "w2", ],
                # target_modules="all-linear",
                task_type='CAUSAL_LM',
                r=16,
                lora_alpha=1,
                lora_dropout=0.1,
            )

            # self.dec_model = AutoModel.from_pretrained("internlm/internlm2_5-step-prover-critic",
            #                                            trust_remote_code=True,
            #                                            torch_dtype=torch.float16, device_map="cuda",
            #                                            config=dec_config, )
            # # quantization_config=quant_config, )

            self.enc_model = get_peft_model(self.enc_model, lora_config)
            self.enc_model.print_trainable_parameters()
            # self.enc_model = self.enc_model.model

            # lora_config = LoraConfig(
            #     target_modules=[
            #         "wqkv",
            #         "wo",
            #         "gate_up_proj",
            #         "w2", ],
            #
            #     task_type='CAUSAL_LM',
            #     r=16,
            #     lora_alpha=1,
            #     lora_dropout=0.1,
            # )
            #
            # self.dec_model = get_peft_model(self.dec_model, lora_config)
            #
            # self.dec_model.output.weight.requires_grad = True
            #
            # self.dec_model.print_trainable_parameters()

            # self.dec_model = self.dec_model.model

            self.error_network = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim // 2),
                torch.nn.LayerNorm(self.emb_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim // 2, 1),
            )

            self.score_network = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim // 2),
                torch.nn.LayerNorm(self.emb_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim // 2, 1),
            )

        # "wqkv",
        # "wo",
        # "gate_up_proj",
        # "w2",
        #
        #     config = LoraConfig(
        #         target_modules=list(config.lora_config.target_modules),
        #         task_type=config.lora_config.task_type,
        #         r=config.lora_config.r,
        #         lora_alpha=config.lora_config.lora_alpha,
        #         lora_dropout=config.lora_config.lora_dropout,
        #     )
        #     self.generator = get_peft_model(generator, config)
        #     logger.info(f"LoRA: ")
        #     self.generator.print_trainable_parameters()

        self.max_seq_len = config.max_length
        self.num_samples = config.num_samples
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps

        self.topk_accuracies = dict()
        for k in range(1, self.num_samples + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

        self.score_weight = config.score_weight
        self.error_weight = config.error_weight
        self.num_skipped = 0
        self.error_skipped = 0
        self.num_errors = 0
        self.num_success = 0
        self.num_oom = 0
        self.label_weights = config.label_weights

        self.ce_loss = CrossEntropyLoss(weight=torch.tensor(self.label_weights))
        self.bcm = BinaryConfusionMatrix(normalize='none')

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    # bottleneck information to single tactic vec
    def get_tac_encoding(self,
                         goal_and_tac_ids: torch.Tensor,
                         goal_and_tac_mask: torch.Tensor,
                         target_inds,
                         ):

        # encode goals with all tactics included
        output = self.enc_model.model.model.forward(goal_and_tac_ids,
                                                    attention_mask=goal_and_tac_mask).last_hidden_state

        # get tokens only for selected tactics
        tac_tokens = [output[i][target_inds[0][i]:target_inds[0][i] + target_inds[1][i] - 1] for i in
                      range(goal_and_tac_ids.shape[0])]

        print ([output[i].shape for i in range(goal_and_tac_ids.shape[0])])

        print ([tac_tokens[i].shape for i in range(goal_and_tac_ids.shape[0])])


        tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-step-prover-critic", trust_remote_code=True)

        print (tokenizer.decode(goal_and_tac_ids[0][target_inds[0][0]:target_inds[0][0] + target_inds[1][0] - 1]))

        # get the tactic embeddings and mean pool them using the provided lengths
        tac_enc = []

        for enc in tac_tokens:
            # mean pool each tactic
            enc = enc.sum(dim=0) / enc.shape[0]
            enc = F.normalize(enc, dim=0)
            tac_enc.append(enc)

        tac_enc = torch.stack(tac_enc, dim=0).unsqueeze(1)

        return tac_enc.squeeze(1)

    # todo could rewrite forward method from InternLMModel to use custom attention mask
    # where tactics cannot attend to each other?
    def forward(
            self,
            goal_and_tac_ids: torch.Tensor,
            goal_and_tac_mask: torch.Tensor,
            goal_ids: torch.Tensor,
            goal_mask: torch.Tensor,
            target_inds: torch.Tensor,
            result_ids: torch.Tensor,
            result_mask: torch.Tensor,
            score_targets: torch.Tensor,
            error_targets: torch.Tensor,
    ):

        if goal_and_tac_ids.shape[-1] >= self.max_seq_len:
            # if goal_and_tac_ids.shape[-1] + result_ids.shape[-1] + 1 >= self.max_seq_len:
            # print('Sequence too long, skipping..')
            self.num_skipped += 1
            return None

        try:
            tac_enc = self.get_tac_encoding(goal_and_tac_ids, goal_and_tac_mask, target_inds)

            # goal_id_lens = goal_mask.sum(dim=1)
            #
            # # get embeddings for goal + outcome
            # combined_ids = torch.cat([goal_ids, result_ids], dim=1)
            #
            # # revert to pad token from -100 HF token
            # combined_ids[combined_ids == -100] = self.dec_model.model.config.pad_token_id
            #
            # # (batch x seq_len x emb_dim)
            # combined_embeds = self.dec_model.model.model.tok_embeddings(combined_ids)
            #
            # # set first embedding to be the tactic encoding
            # combined_embeds = torch.cat([tac_enc, combined_embeds], dim=1)
            #
            # new_mask = torch.cat([torch.ones(goal_mask.shape[0], 1).to(self.device), goal_mask, result_mask], dim=1)
            #
            # # (batch x seq_len x vocab_size)
            #
            # logits = self.dec_model.model.forward(inputs_embeds=combined_embeds, attention_mask=new_mask).logits
            #
            # # only consider labels from goal_ids_lens (plus tactic) onwards (i.e. result_ids)
            # # todo only works for batch_size == 1, not clear how to do this for multiple elements
            #
            # logits = logits[:, goal_id_lens[0] + 1:, :]
            #
            # # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            #
            # shift_labels = result_ids[..., 1:].contiguous()
            #
            # # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, logits.shape[-1])
            # shift_labels = shift_labels.view(-1)
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            #
            # dec_loss = loss_fct(shift_logits, shift_labels)
            #
            # batch_size x 1 (score_prediction)
            score_output = self.score_network(tac_enc)  # .squeeze(1)

            score_loss = F.mse_loss(score_output, score_targets)

            error_output = self.error_network(tac_enc)#.squeeze(-1).squeeze(-1)

            error_preds = torch.sigmoid(error_output[:,0])
            error_preds = error_preds.unsqueeze(1)
            error_preds = torch.cat([1 - error_preds, error_preds], dim=1)

            error_loss = self.ce_loss(error_preds, error_targets)

            # return score_loss, dec_loss, error_loss
            return score_loss, error_loss

        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.num_oom += 1
            self.num_skipped += 1
            return None

    def backward(self, loss, *args, **kwargs):
        try:
            super().backward(loss, *args, **kwargs)
        except RuntimeError:
            torch.cuda.empty_cache()
            self.num_oom += 1
            self.num_skipped += 1
            return None

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        error_targets = torch.tensor(
            [1 if batch['status'][i] == 'success\n' else 0 for i in range(len(batch['status']))],
            dtype=torch.long).to(self.device)

        losses = self(
            batch['goal_and_tac_ids'],
            batch['goal_and_tac_mask'],
            batch["goal_ids"],
            batch["goal_mask"],
            batch["target_inds"],
            batch["result_ids"],
            batch['result_mask'],
            batch['score_targets'],
            error_targets
        )
        if batch['status'][0] != 'success\n':
            self.num_errors += 1
        else:
            self.num_success += 1

        if losses:
            # score_loss, dec_loss, error_loss = losses
            score_loss, error_loss = losses
        else:
            if batch['status'][0] != 'success\n':
                self.error_skipped += 1
            return None
        #
        # self.log(
        #     "dec_loss_train",
        #     dec_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=len(batch),
        #     prog_bar=True
        # )

        self.log(
            "error_loss_train",
            error_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "oom",
            self.num_oom / (self.num_success + self.num_errors),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )
        self.log(
            "skip",
            self.num_skipped / (self.num_success + self.num_errors),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )
        #
        # self.log(
        #     "s_skip",
        #     (self.num_skipped - self.error_skipped) /  self.num_success,
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=len(batch),
        #     prog_bar=True
        # )
        #
        # self.log(
        #     "e_skip",
        #     self.error_skipped / self.num_errors,
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=len(batch),
        #     prog_bar=True
        # )

        # self.log(
        #     "num_errors",
        #     self.num_errors / (self.num_success + self.num_errors),
        #     on_step=True,
        #     on_epoch=True,
        #     sync_dist=True,
        #     batch_size=len(batch),
        #     prog_bar=True
        # )

        # ignore score prediction for error tactics
        if batch['status'][0] != 'success\n':
            # return dec_loss + self.error_weight * error_loss

            return  self.error_weight * error_loss
        else:
            self.log(
                "score_loss_train",
                score_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
                prog_bar=True
            )
            return self.score_weight * score_loss + self.error_weight * error_loss

    ##############
    # Validation #
    ##############

    # def on_validation_epoch_start(self) -> None:
    #     if self.global_rank == 0:
    #         # using columns and data
    #         self.log_table = []
    #
    # def on_validation_epoch_end(self) -> None:
    #     if self.global_rank == 0:
    #         self.logger.log_table(key=f'predictions_{self.global_step}',
    #                               columns=["goal", "tactic",
    #                                        "score",
    #                                        "score_prediction",
    #                                        "error_prediction",
    #                                        "error_probs",
    #                                        ],
    #                               data=self.log_table)
    #
    def validation_step(self, batch: Dict[str, Any], _) -> None:
        try:

            error_targets = torch.tensor(
                [1 if batch['status'][i] == 'success\n' else 0 for i in range(len(batch['status']))],
                dtype=torch.long).to(self.device)

            tac_enc = self.get_tac_encoding(batch['goal_and_tac_ids'], batch['goal_and_tac_mask'], batch['target_inds'])

            score_output = self.score_network(tac_enc)  # .squeeze(1)

            score_loss = F.mse_loss(score_output, batch['score_targets'])

            error_output = self.error_network(tac_enc)#.squeeze(-1)# .squeeze(-1)

            error_preds = torch.sigmoid(error_output[:, 0])

            # get preds as those > 0.5
            error_preds = error_preds > 0.5
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

            self.log(
                "score_loss_val",
                score_loss,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
                prog_bar=False,
            )

            # if self.global_rank == 0:
            #     data = [[batch['goal'][i], batch['tactic'][i], batch['score_targets'][i], batch['time_targets'][i],
            #              nl.join(predictions[i]), 'success' if error_preds[i] == 1 else 'failure', error_probs[i],
            #              time_preds[i]]
            #             for i in range(batch_size)]
            #     self.log_table.extend(data)
            return
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.num_oom += 1
            self.num_skipped += 1
            return None


