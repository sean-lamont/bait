"""Data module for the tactic generator."""
import math
import pickle
from typing import Optional

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.common import (
    Batch,
)
from experiments.end_to_end.process_traces import add_rand_idx, filter_traces
from experiments.end_to_end.proof_node import ErrorNode
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn


# new ranking:
# - restricted to max_pairs per goal
# - prioritise rankings: (proven, error), (no error, error)
# - try find pairs with closer length
def rank_edges_new(goal, edges, max_pairs=16):
    valid_edges = [edge for edge in edges if not isinstance(edge.dst[0], ErrorNode)]
    invalid_edges = [('error', edge) for edge in edges if isinstance(edge.dst[0], ErrorNode)]

    # proven edges
    proven_edges = [('proven', edge) for edge in valid_edges if edge.distance_to_proof() < math.inf]

    # non-error edges
    success_non_proven_edges = [('no_error', edge) for edge in valid_edges if edge.distance_to_proof() == math.inf]

    all_edges = sorted(invalid_edges + proven_edges + success_non_proven_edges, key=lambda x: len(x[1].tactic))

    num_proven = len(proven_edges)
    num_failed = len(invalid_edges)
    num_success = len(success_non_proven_edges)

    pairs = []

    while len(pairs) < max_pairs and num_failed >= 0:
        winner = None
        loser = None
        win_type = None
        last_error = None
        for (i, (edge_type, edge)) in enumerate(all_edges):
            if not winner:
                if edge_type == 'proven':
                    winner = edge
                    all_edges.pop(i)
                    num_proven -= 1
                    win_type = edge_type
                elif num_proven <= 0 and edge_type == 'no_error':
                    winner = edge
                    all_edges.pop(i)
                    num_success -= 1
                    win_type = edge_type
                elif edge_type == 'error':
                    last_error = edge

            # will be called once winner is found
            elif edge_type == 'error':
                # nearest error edge will be either last_error or this error, take the closest in tactic length
                if not last_error:
                    loser = edge
                    num_failed -= 1
                    all_edges.pop(i)
                    break
                elif (len(last_error.tactic) - len(winner.tactic)) ** 2 <= (len(edge.tactic) - len(edge.tactic)) ** 2:
                    loser = last_error
                    num_failed -= 1
                    all_edges.pop(i)
                    break
                else:
                    loser = edge
                    num_failed -= 1
                    all_edges.pop(i)
                    break

        if winner and loser and win_type:
            pairs.append((winner, loser, win_type))
        else:
            return

        w_l = [
            {'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic,
             'loser_prob': l.tac_logprob,
             'type': tac_type} for (w, l, tac_type) in pairs]

        return w_l


def rank_edges(goal, edges):
    valid_edges = [edge for edge in edges if not isinstance(edge.dst[0], ErrorNode)]
    invalid_edges = [edge for edge in edges if isinstance(edge.dst[0], ErrorNode)]

    # rank all valid_edges above all invalid_edges
    w_l = [
        {'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic, 'loser_prob': l.tac_logprob,
         'type': 'valid_rank'} for w in valid_edges for l in invalid_edges]

    # from valid_edges, rank proven goals above non_proven valid goals
    proven_edges = [edge for edge in valid_edges if edge.distance_to_proof() < math.inf]
    success_non_proven_edges = [edge for edge in valid_edges if edge.distance_to_proof() == math.inf]

    w_l.extend([{'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic,
                 'loser_prob': l.tac_logprob,
                 'type': 'proven_rank'} for w in proven_edges for l in success_non_proven_edges])

    # from proven edges, rank based on distance_to_proof, then execution time
    ranked_proofs = sorted(proven_edges, key=lambda x: (x.distance_to_proof(), x.time))

    w_l.extend(
        [{'goal': goal, 'winner': ranked_proofs[i].tactic,
          'winner_prob': ranked_proofs[i].tac_logprob, 'loser': ranked_proofs[j].tactic,
          'loser_prob': ranked_proofs[j].tac_logprob,
          'type': 'time_len_rank'} for i in range(len(ranked_proofs)) for j in
         range(i + 1, len(ranked_proofs))])

    # among successful without proof, rank those that lead to the same outcome based on time only
    for i, edge in enumerate(success_non_proven_edges):
        same_outcome_ranks = []
        for j in range((i + 1), len(success_non_proven_edges)):
            edge_2 = success_non_proven_edges[j]
            edge_1_outcome = [g.goal for g in edge.dst] if isinstance(edge.dst[0], InternalNode) else [
                'Error'] if isinstance(edge.dst[0], ErrorNode) else ['Proven']
            edge_2_outcome = [g.goal for g in edge_2.dst] if isinstance(edge_2.dst[0], InternalNode) else [
                'Error'] if isinstance(edge_2.dst[0], ErrorNode) else ['Proven']
            if set(edge_1_outcome) == set(edge_2_outcome):
                if edge.time < edge_2.time:
                    same_outcome_ranks.append(
                        {'goal': goal, 'winner': edge.tactic, 'winner_prob': edge.tac_logprob, 'loser': edge_2.tactic,
                         'loser_prob': edge_2.tac_logprob, 'type': 'same_outcome'})
                else:
                    same_outcome_ranks.append(
                        {'goal': goal, 'winner': edge_2.tactic, 'winner_prob': edge_2.tac_logprob, 'loser': edge.tactic,
                         'loser_prob': edge.tac_logprob, 'type': 'same_outcome'})

        w_l.extend(same_outcome_ranks)

    return w_l


class DPODataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            trace_files=None,
            database='lean_e2e',
            collection='dpo'
    ) -> None:

        super().__init__()

        if trace_files is None:
            trace_files = []
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["goal", "winner", "winner_prob", "loser", "loser_prob"]
        self.collection = collection
        self.database = database
        self.current_train_batch_index = 0
        self.trace_files = trace_files

    def state_dict(self):
        self.current_train_batch_index = self.ds_train.start_idx
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
        self.setup()

    def prepare_data(self):
        trace_files = filter_traces(self.trace_files)
        traces = []

        for file in trace_files:
            with open(file, 'rb') as f:
                traces.append(pickle.load(f))

        if not traces:
            return

        logger.info('Processing traces for DPO model...')

        collection = MongoClient()[self.database][self.collection]

        for trace in tqdm(traces):
            if isinstance(trace.tree, ErrorNode):
                continue

            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            # add edge ranking data for DPO
            for node in nodes.values():
                if node.out_edges:
                    # select which ranking approach here
                    w_l = rank_edges_new(goal=node.goal, edges=node.out_edges)

                    if w_l:
                        collection.insert_many(w_l)

        add_rand_idx(collection)

    def setup(self, stage: Optional[str] = None) -> None:
        # 90/10 train/val ratio
        train_range = (0., 0.95)
        val_range = (0.95, 1.)

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              range=(train_range[0], train_range[1]),
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              start_idx=self.current_train_batch_index
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
        goals = [g['goal'] for g in examples]
        winners = [g['winner'] for g in examples]
        winner_probs = [g['winner_prob'] for g in examples]
        losers = [g['loser'] for g in examples]
        loser_probs = [g['loser_prob'] for g in examples]

        tokenized_state = self.tokenizer(
            goals,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_winners = self.tokenizer(
            winners,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_losers = self.tokenizer(
            losers,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        winner_ids = tokenized_winners.input_ids
        winner_ids[winner_ids == self.tokenizer.pad_token_id] = -100

        loser_ids = tokenized_losers.input_ids
        loser_ids[loser_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": goals,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "winners": winners,
                 "winner_ids": winner_ids,
                 "winner_attention_mask": tokenized_winners.attention_mask,
                 "losers": losers,
                 "loser_ids": loser_ids,
                 "loser_attention_mask": tokenized_losers.attention_mask,
                 "winner_ref_probs": torch.FloatTensor(winner_probs),
                 "loser_ref_probs": torch.FloatTensor(loser_probs)
                 }

        return batch
