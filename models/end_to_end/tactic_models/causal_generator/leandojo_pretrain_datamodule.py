"""Data module for the tactic generator."""
import copy
import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
import lightning.pytorch as pl
from typing import Optional, List, Dict, Any
from lean_dojo.constants import LEAN3_PACKAGES_DIR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer
from trl import DataCollatorForCompletionOnlyLM

from experiments.end_to_end.common import (
    Batch,
    Corpus,
    Example,
    format_state,
    remove_marks,
    format_tactic,
    format_augmented_state, LeanDojoCorpus,
)


class GeneratorDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            corpus: Corpus,
            keep_marks: bool,
            preds: List[Dict[str, Any]],
            max_seq_len: int,
            p_drop: float,
            normalize_tactics: bool,
            tokenizer: Any,
            is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.keep_marks = keep_marks
        self.preds = preds
        self.max_seq_len = max_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path, normalize_tactics)


        ## specific to LLAMA based tokeniser, ensure that the collator response template has context
        response_template_with_context = "\n[ANSWER]"  # We added context here: "\n". This is enough for this tokenizer
        response_template_ids = self.tokenizer.encode(response_template_with_context, add_special_tokens=False)[
                                2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

        self.collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False,
                                                        return_tensors="pt")

    def _load_data(self, data_path: str, normalize_tactics: bool) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                if "annotated_tactic" in tac:
                    tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
                else:
                    tactic = format_tactic(tac["tactic"], [], normalize_tactics)
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": format_state(tac["state_before"]),
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = copy.deepcopy(self.data[idx])

        if self.preds is not None:
            if ex["file_path"] in self.corpus:
                file_path = ex["file_path"]
            else:
                _, repo_name = os.path.split(ex["url"])
                file_path = os.path.join(LEAN3_PACKAGES_DIR, repo_name, ex["file_path"])

            pred = self.preds[(file_path, ex["full_name"], ex["state"])]

            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_seq_len,
                self.p_drop if self.is_train else 0.0,
            )

        if not self.keep_marks:
            ex["state"] = remove_marks(ex["state"])

        return ex

    # need to have same input/output shape for labels with causal LM
    def collate(self, examples: List[Example]) -> Batch:
        
        
        prompt = ('You are an expert in Lean 3 theorem proving.'
                  'Given a set of premises, followed by a goal to prove, suggest a single tactic to solve the goal.'
                  'Any premises in the tactic should be included in the following format: <a>premise<\\a>. The goal is: \n\n')

        state = [prompt + ex["state"] + '[ANSWER]' + ex["tactic"] + self.tokenizer.eos_token for ex in examples]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )


        collated = self.collator(list(tokenized_state.input_ids))
        state_ids = collated['input_ids']

        tactic_ids = collated['labels']
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = state_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic_ids"] = tactic_ids
        # batch["tactic_mask"] = tokenized_tactic.attention_mask

        # print (batch)
        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            keep_marks: bool,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            p_drop: float,
            normalize_tactics: bool,
            num_workers: int,
            corpus_path: Optional[str] = None,
            preds_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        if corpus_path is not None:
            self.corpus = LeanDojoCorpus(corpus_path)
        else:
            self.corpus = None
        self.keep_marks = keep_marks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.p_drop = p_drop
        self.normalize_tactics = normalize_tactics
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if preds_path is None:
            logger.info("Without retrieval data")
            self.preds = None
        else:
            logger.info("With retrieval data")
            self.preds = {}
            for pred in pickle.load(open(preds_path, "rb")):
                ctx = pred["context"]
                self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                os.path.join(self.data_path, "train.json"),
                self.corpus,
                self.keep_marks,
                self.preds,
                self.max_seq_len,
                self.p_drop,
                self.normalize_tactics,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                os.path.join(self.data_path, "val.json"),
                self.corpus,
                self.keep_marks,
                self.preds,
                self.max_seq_len,
                self.p_drop,
                self.normalize_tactics,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
