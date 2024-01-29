"""Lightning module for the tactic generator."""

import re
from subprocess import CalledProcessError
from typing import Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lean_dojo.utils import execute
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration, AutoTokenizer

from experiments.end_to_end.common import format_augmented_state, zip_strict, get_optimizers, load_checkpoint
from models.end_to_end.tactic_models.retrieval.model import PremiseRetriever

torch.set_float32_matmul_precision("medium")


class GenTacModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.gen_config = config.gen_config
        self.eval_config = config.eval_config
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps
        self.max_seq_len = config.max_seq_len

        # todo more general
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        generator = T5ForConditionalGeneration.from_pretrained(config.model_name)

        ret_ckpt_path = config.ret_ckpt_path if hasattr(config, 'ret_ckpt_path') else None

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            self.retriever = PremiseRetriever.load(
                ret_ckpt_path, self.device, freeze=True
            )

        self.live_eval = config.live_eval if hasattr(config, 'live_eval') else False

        if hasattr(config, 'lora_config') and config.lora_config:
            config = LoraConfig(
                target_modules=list(config.lora_config.target_modules),
                task_type=config.lora_config.task_type,
                r=config.lora_config.r,
                lora_alpha=config.lora_config.lora_alpha,
                lora_dropout=config.lora_config.lora_dropout,
            )
            self.generator = get_peft_model(generator, config)
            logger.info(f"LoRA: ")
            self.generator.print_trainable_parameters()
        else:
            self.generator = generator

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def on_fit_start(self) -> None:
        if self.logger is not None and self.global_rank == 0:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    ###############################
    # Evaluation with live prover #
    ###############################

    def on_validation_epoch_end(self) -> None:
        if self.global_step > 1 and self.live_eval:
            torch.cuda.empty_cache()
            self.run_eval()

    def run_eval(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last_eval.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        # todo get config file from config
        cmd = f"python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/leandojo num_theorems={self.eval_config.eval_num_theorems}" \
              f" shuffle={self.eval_config.shuffle} env_timeout={self.eval_config.timeout} tac_model.ckpt_path={ckpt_path} log_level='ERROR' tac_model.model='dpo'" \
              f" exp_config.name=eval_{self.global_step} exp_config.experiment=dpo_eval"

        logger.info(f'Running evaluation with {cmd}')

        try:
            # todo better/live output?:
            #  https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
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

    ##############
    # Prediction #
    ##############

    def generate(self, state: str, retriever_args: dict, num_samples: int):
        return self.batch_generate([state], retriever_args, num_samples)[0]

    def batch_generate(self, state, retriever_args, num_samples):
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                retriever_args,
                self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        if self.gen_config.strategy == 'sample':
            return self.sample_gen(state, state_ids, state_mask, num_samples)
        elif self.gen_config.strategy == 'beam':
            return self.beamsearch_gen(state, state_ids, state_mask, num_samples)

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
                num_return_sequences=num_samples * 2,
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
            early_stopping=False,
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