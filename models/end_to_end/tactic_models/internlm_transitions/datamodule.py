"""Data module for the tactic generator."""
import copy
import math
import pickle
from pathlib import Path
import random
from typing import Optional

import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.lightning_common import Batch
from experiments.end_to_end.process_traces import add_rand_idx, filter_traces
from experiments.end_to_end.proof_node import ErrorNode, Status
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn

'''


'''


def get_prev_tactics(goal):
    if not goal.in_edges:
        return ''
    else:
        return get_prev_tactics(goal.in_edges[0].src) + goal.in_edges[0].tactic


# todo using critic model trace, get scores, task is to predict scores and state transition
class InternLMTransitionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            trace_files=None,
            database='internlm_transition',
            collection='critic_batched_tactics',
            replace='keep',
            host='localhost:27017'  # mongodb host
    ) -> None:

        super().__init__()

        if trace_files is None:
            trace_files = []
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ['goal', 'tactic', 'result', 'theorem', 'status', 'goal_score', 'all_tacs', 'tac_index']
        self.collection = collection
        self.database = database
        self.current_train_batch_index = 0
        self.trace_files = trace_files
        self.replace = replace
        self.host = host

    def state_dict(self):
        self.current_train_batch_index = self.ds_train.start_idx
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
        self.setup()

    def prepare_data(self):
        db = MongoClient(self.host)[self.database]

        if self.collection in db.list_collection_names():
            if self.replace == 'keep':
                logger.info('Collection exists, skipping.')
                return
            elif self.replace == 'add':
                logger.info('Collection exists, adding to.')
            elif self.replace == 'drop':
                logger.info('Collection exists, dropping.')
                db[self.collection].drop()
            else:
                raise ValueError(f'Invalid value for replace: {self.replace}')

        path = Path(self.trace_files)
        trace_files = [x for x in path.rglob("*") if x.is_file()]

        if not trace_files:
            return

        collection = MongoClient()[self.database][self.collection]

        def add_trace(trace, split_size):
            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            print (list(nodes.values())[0].out_edges[0])

            for node in nodes.values():
                if node.out_edges:
                    split = 'train' if random.random() < split_size else 'val'
                    state = f"What is the result of applying TACTICS to STATE? ---\nNAME: {trace.theorem.full_name}\n\n---\nPROOF_BEFORE: {get_prev_tactics(node)}\n\n---\nSTATE: {node.goal}\n\n---\nTACTICS: \n\n"
                    data = {'goal': state, 'theorem': trace.theorem.full_name, 'split': split}
                    transitions = []
                    seen_tacs = set()
                    for edge in node.out_edges:
                        tac = edge.tactic
                        if tac not in seen_tacs:
                            seen_tacs.add(tac)
                            if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
                                result = edge.dst[0].inner.message.split(' tactic_state')[0]
                                status = 'failed\n'
                            else:
                                result = ''.join([d.goal if hasattr(d, 'goal') else 'no goals' for d in edge.dst])
                                status = 'success\n'
                            # if edge.goal_logprob > -math.inf:
                                # print (edge, edge.goal_logprob)
                            transitions.append({'tactic': tac, 'result': result, 'status': status, 'goal_score': edge.goal_logprob})

                    # all_tactics = "\n".join([t['tactic'] for t in transitions])
                    all_tactics = [t['tactic'] for t in transitions]


                    for i, t in enumerate(transitions):
                        data[f'tactic'] = t['tactic']
                        data[f'result'] = t['result']
                        data[f'status'] = t['status']
                        data[f'goal_score'] = t['goal_score']
                        data[f'tac_index'] = i
                        data[
                            f'all_tacs'] = all_tactics  # give all tactics to the model, but only extract embedding for given index
                        collection.insert_one(copy.deepcopy(data))

        logger.info('Processing traces for training transition model...')
        for trace in tqdm(trace_files):
            try:
                trace = pickle.load(open(trace, 'rb'))
            except Exception as e:
                logger.info(f'Error loading {trace}: {e}')
                continue
            if isinstance(trace.tree, ErrorNode):
                continue

            add_trace(trace, 0.95)

        add_rand_idx(collection)

    def setup(self, stage: Optional[str] = None) -> None:
        train_filter = [{'$match': {'split': 'train'}},
                        {'$sort': {'rand_idx': 1}}]

        val_filter = [{'$match': {'split': 'val'}},
                      {'$sort': {'rand_idx': 1}}]

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              filter_=train_filter,
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              host=self.host
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            filter_=val_filter,
                                            gpu_id=self.trainer.global_rank,
                                            num_gpus=self.trainer.num_devices,
                                            host=self.host
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
        # goal = [ex["theorem"] + '\n\n' + ex["goal"][int(len(ex["goal"]) * 0.35):] for ex in examples]
        # goal = [ex["tactic"] + ex["theorem"] + '\n\n' + ex["goal"][int(len(ex["goal"]) * 0.6):] for ex in examples]
        # goal = [ex["tactic"] + ex["theorem"] + '\n\n' + ex["goal"] for ex in examples]


        # tokenise goal and tactics up to target tactic, then target tactic, take indices based on this

        tokenised_up_to_target = [ex['state'] + '\n'.join([t for t in ex['all_tactics'][:ex['tac_index']]]) for ex in examples]

        # todo get location of tactic tokens in tokenised goal

        tokenised_up_to_target = self.tokenizer(tokenised_up_to_target,
                                                padding=None,
                                                max_length =self.max_seq_len,
                                                truncation=True, return_tensors='pt')


        lens_before = tokenised_up_to_target.attention_mask.sum(dim=1)

        target_tactics = ['\n' + ex['tactic'] for ex in examples]

        tokenized_tactics = self.tokenizer(
            target_tactics,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tac_lens = tokenized_tactics.attention_mask.sum(dim=1)

        target_inds = (lens_before, tac_lens)


        goal = [ex['state'] + '\n'.join(ex['all_tactics']) for ex in examples]

        tokenized_goal = self.tokenizer(
            goal,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        result = [ex['status'] + ex["result"] for ex in examples]

        tokenized_result = self.tokenizer(
            result,
            padding="longest",
            max_length=self.max_seq_len - tokenized_goal.input_ids.shape[1],
            truncation=True,
            return_tensors="pt",
        )

        # print (tokenized_goal.input_ids.shape, tokenized_result.input_ids.shape)


        # result_ids = tokenized_result.input_ids

        # result_ids[result_ids == self.tokenizer.pad_token_id] = -100  # todo equivalent token id for internlm?

        batch = {}
        batch["goal"] = goal
        batch["goal_ids"] = tokenized_goal.input_ids
        batch["goal_mask"] = tokenized_goal.attention_mask
        batch["result"] = result
        batch["result_ids"] = tokenized_result.input_ids
        batch["result_mask"] = tokenized_goal.attention_mask
        batch["tactic"] = target_tactics
        batch["target_inds"] = target_inds
        batch["status"] = [ex['status'] for ex in examples]

        # # Copy other fields.
        # for k in examples[0].keys():
        #     if k not in batch:
        #         batch[k] = [ex[k] for ex in examples]

        return batch
