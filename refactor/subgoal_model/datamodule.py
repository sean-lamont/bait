"""Data module for the tactic generator."""
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from refactor.common import (
    Batch,
)
from refactor.dpo.datamodule import GoalStreamDataset, worker_init_fn


class SubgoalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            critic_tok: str,
            separation_tok: str,
            provable_tok: str,
            unprovable_tok: str,
            database='lean_e23',
            collection='goal_proven_task'
    ) -> None:

        super().__init__()

        # conditions model on the task
        self.critic_tok = critic_tok
        # token to separate initial and subgoal
        self.separation_tok = separation_tok

        self.provable_tok = provable_tok
        self.unprovable_tok = unprovable_tok

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["init_goal", "subgoal", "target"]
        self.collection = collection
        self.database = database

        # self.test_col = MongoClient()['lean_dojo']['test_col']

    def setup(self, stage: Optional[str] = None) -> None:
        # 90/10 train/val ratio
        train_range = (0., 0.9)
        val_range = (0.9, 1.)

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              range=(train_range[0], train_range[1]),
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            range=(val_range[0], val_range[1]),
                                            gpu_id=self.trainer.global_rank,
                                            num_gpus=self.trainer.num_devices,
                                            )

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          drop_last=True,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True
                          )

    def collate_fn(self, examples) -> Batch:
        subgoals = [g['goal'] for g in examples]
        initial_goals = [g['initial_goal'] for g in examples]
        targets = [g['target'] for g in examples]

        # concat the special tokens and Initial / Subgoals
        state = [self.critic_tok + initial_goals[i] + self.separation_tok + subgoals[i] for i in range(len(subgoals))]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        targets = [self.provable_tok if ex == 1 else self.unprovable_tok for ex in targets]

        tokenized_target = self.tokenizer(
            targets,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # values set to -100 ignored in HuggingFace loss
        target_ids = tokenized_target.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": state,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "targets": targets,
                 "target_ids": target_ids,
                 "target_attention_mask": tokenized_target.attention_mask}

        return batch