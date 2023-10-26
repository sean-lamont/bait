"""Proof search using best-first search.
"""
import heapq
import os
import pickle
import sys
import time
from typing import Dict, Tuple, Any

import ray
import torch
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    ProofFinished,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
from lean_dojo.constants import LEAN3_DEPS_DIR, LEAN4_DEPS_DIR
from ray.util.actor_pool import ActorPool

from experiments.reprover.common import zip_strict
from experiments.reprover.generator.model import RetrievalAugmentedGenerator
from experiments.reprover.goal_model_step.model import StepGoalModel
from refactor.proof_node import *

from loguru import logger


@dataclass
class AbstractTheorem:
    # todo should be union of Lean Theorem, HOL4 Theorem, HOList Theorem
    theorem: Any

@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: AbstractTheorem
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)

    # Some statistics during proof search.
    stats: Dict





# @ray.remote
# class CpuProver(BestFirstSearchProver):
#     """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""
#
#     def __init__(
#             self,
#             ckpt_path: str,
#             goal_path: str,
#             indexed_corpus_path: Optional[str],
#             timeout: int,
#             num_sampled_tactics: int,
#             debug: bool,
#     ) -> None:
#
#         tac_gen = RetrievalAugmentedGenerator.load(
#             ckpt_path, device=torch.device("cpu"), freeze=True
#         )
#
#         if tac_gen.retriever is not None:
#             if indexed_corpus_path is not None:
#                 tac_gen.retriever.load_corpus(indexed_corpus_path)
#             tac_gen.retriever.reindex_corpus(batch_size=32)
#
#         goal_model = StepGoalModel.load(goal_path, device=torch.device("cpu"), freeze=True)
#
#         super().__init__(
#             tac_gen,
#             goal_model,
#             timeout,
#             num_sampled_tactics,
#             debug,
#         )
#
#
# # set to 1 / (gpu_mem //  required_mem)
# @ray.remote(num_gpus=1)
# class GpuProver(BestFirstSearchProver):
#     """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""
#
#     def __init__(
#             self,
#             ckpt_path: str,
#             goal_path: str,
#             indexed_corpus_path: Optional[str],
#             timeout: int,
#             num_sampled_tactics: int,
#             debug: bool,
#     ) -> None:
#
#         tac_gen = RetrievalAugmentedGenerator.load(
#             ckpt_path, device=torch.device("cuda"), freeze=True
#         )
#
#         if tac_gen.retriever is not None:
#             if indexed_corpus_path is not None:
#                 tac_gen.retriever.load_corpus(indexed_corpus_path)
#             tac_gen.retriever.reindex_corpus(batch_size=32)
#
#         goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)
#
#         super().__init__(
#             tac_gen,
#             goal_model,
#             timeout,
#             num_sampled_tactics,
#             debug,
#         )
#
#
# class DistributedProver:
#     """A distributed prover that uses Ray to parallelize the proof search.
#
#     It is a wrapper around `CpuProver` and `GpuProver` that handles the different
#     devices and different number of concurrent provers.
#     """
#
#     def __init__(
#             self,
#             ckpt_path: str,
#             goal_path: str,
#             indexed_corpus_path: Optional[str],
#             num_cpus: int,
#             with_gpus: bool,
#             timeout: int,
#             num_sampled_tactics: int,
#             debug: Optional[bool] = False,
#     ) -> None:
#         self.distributed = num_cpus > 1
#         if not self.distributed:
#             device = torch.device("cuda") if with_gpus else torch.device("cpu")
#             tac_gen = RetrievalAugmentedGenerator.load(
#                 ckpt_path, device=device, freeze=True
#             )
#             if tac_gen.retriever is not None:
#                 assert indexed_corpus_path is not None
#                 tac_gen.retriever.load_corpus(indexed_corpus_path)
#             # goal_model = None
#
#             goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)
#
#             self.prover = BestFirstSearchProver(
#                 tac_gen, goal_model, timeout, num_sampled_tactics, debug
#                 # tac_gen, goal_model, timeout, num_sampled_tactics, debug
#             )
#             return
#
#         if with_gpus:
#             logger.info(f"Launching {num_cpus} GPU workers.")
#             ray.init(num_cpus=num_cpus, num_gpus=num_cpus)
#             provers = [
#                 GpuProver.remote(
#                     ckpt_path,
#                     goal_path,
#                     indexed_corpus_path,
#                     timeout=timeout,
#                     num_sampled_tactics=num_sampled_tactics,
#                     debug=debug,
#                 )
#                 for _ in range(num_cpus)
#             ]
#         else:
#             logger.info(f"Launching {num_cpus} CPU workers.")
#             ray.init(num_cpus=num_cpus, num_gpus=0)
#             provers = [
#                 CpuProver.remote(
#                     ckpt_path,
#                     goal_path,
#                     indexed_corpus_path,
#                     timeout=timeout,
#                     num_sampled_tactics=num_sampled_tactics,
#                     debug=debug,
#                 )
#                 for _ in range(num_cpus)
#             ]
#
#         self.prover_pool = ActorPool(provers)
#
#     def search_unordered(
#             self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
#     ) -> List[SearchResult]:
#         # theorems = theorems[120+96:]
#         # positions = positions[120+96:]
#         """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
#         if not self.distributed:
#             return [
#                 self.prover.search(repo, thm, pos)
#                 for thm, pos in zip_strict(theorems, positions)
#             ]
#
#         try:
#             results = list(
#                 self.prover_pool.map_unordered(
#                     lambda p, x: p.search.remote(repo, x[0], x[1]),
#                     zip_strict(theorems, positions),
#                 )
#             )
#         except ray.exceptions.RayActorError as ex:
#             logger.error(ex)
#             sys.exit(1)
#
#         return results
#


# todo make lightning module?
# have main search as e.g. validation dataset, then training data can be defined separately in datamodule to use e.g. logs
# validation data would just be e.g. goals to prove. Datamodule can also give the model the environment
# for a eval step, data model provides: goal, premises, environment
class Search:
    def __init__(self, root_goal, timeout, env):
        # Generic function mapping from state to a tactic
        # self.tac_model = tac_model
        # self.goal_model = goal_model
        self.timeout = timeout
        self.search_stats = {}
        self.env = env

        # will be a ProofNode
        self.root = InternalNode(
            goal=root_goal,
            cumulative_logprob=0.0,
        )

        # Dictionary of strings of goals/states mapping to the corresponding ProofNode
        self.nodes = {self.root.goal: self.root}

    @abstractmethod
    def _get_goals(self):
        return

    @abstractmethod
    def _process_response(self, goals, tactics, response):
        return

    @abstractmethod
    def _process_trace(self, trace):
        return

    @abstractmethod
    def _get_tactics(self, trace):
        return

    def _step(self):
        goals = self.search.get_goals()
        tactics = self.tac_model._get_tactics(goals)
        response = self.env.run(goals, tactics)
        self.search.process_response(goals, tactics, response)

    def search(self, ):
        with torch.no_grad():
            try:
                self._search()
            except Exception as e:
                logger.warning(f"Error in search {e}")
                pass

        if self.root.status == Status.PROVED:
            proof = [e.tactic for e in self.root.extract_proof()]
        else:
            proof = None

        # todo general SearchResult
        result = SearchResult()

        self._process_trace(result)

        return result

    def _search(self) -> None:
        time_start = time.monotonic()
        while True:
            try:
                self._step()
            except Exception as e:
                assert time.monotonic() - time_start >= self.timeout, f"Exception not timeout: {e}"

            self.total_time = time.monotonic() - time_start

            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break


class UpDown(Search):
    def __init__(self, root, tac_model, goal_model, timeout, env):
        super().__init__(root, timeout, env)
        self.tac_model = tac_model
        self.goal_model = goal_model

        self.remaining_tacs = {}
        self.search_stats['goal_time'] = 0
        self.theorem =

    def _get_goals(self):
        if len(self.nodes) == 1:
            return self.root, self.root.provable_score

        best_score = -math.inf
        best_node = None

        for goal, node in self.nodes.items():
            if node.is_explored:
                continue
            # take the score for a node as the probability of proving that goal,
            # multiplied by the probability of proving the best context of that goal
            # (i.e how likely to prove the original goal, assuming this goal is used)
            if node.context and len(node.context[0]) > 0:
                score = node.provable_score + max(
                    [sum([self.nodes[ctx].up_score for ctx in context]) for context in node.context])
            else:
                score = node.provable_score

            if score > best_score:
                best_score = score
                best_node = node

        return best_node, best_score

    # todo move this to tree?
    def _up_step(self, node):
        if node.out_edges:
            best_score = -math.inf
            for edge in node.out_edges:
                edge_score = 0
                for sib in edge.dst:
                    edge_score += sib.up_score

                if edge_score > best_score:
                    best_score = edge_score

            if best_score > node.up_score:
                node.up_score = best_score
                parents = set([edge.src for edge in node.in_edges])
                for parent in parents:
                    self._up_step(parent)

    # Assume response is a single edge in this case
    def _process_response(self, goals, tactics, response):
        search_node = goals

        result = response.dst

        # find new nodes from response, and compute their provable score
        new_nodes = []
        for result_node in result:
            if result_node.goal not in self.nodes:
                new_nodes.append(result_node)

        if new_nodes:
            # compute provable_score/up_score for new internal nodes
            node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]

            t1 = time.monotonic()
            scores = self.goal_model.batch_generate(node_goals)
            self.stats['goal_time'] += time.monotonic() - t1

            for i, node_ in enumerate(new_nodes):
                node_.provable_score = scores[i] + (node_.depth * math.log(0.95))
                node_.up_score = node_.provable_score

        for result_node in result:
            # Record the new node and add it to the search queue.
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(response)
                self.nodes[result_node.goal] = result_node
                search_node.children = search_node.children | {result_node.goal}

        if search_node.out_edges:
            search_node.out_edges = search_node.out_edges + [response]
        else:
            search_node.out_edges = [response]

        self._up_step(search_node)

        self.num_expansions += 1

        # todo add data to trace (goal, tactic, logprob)

        return

    def _process_trace(self, trace):
        # todo save to file/mongodb
        return

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        premises = self.env.retrieve_premises(ts)
        tactics = self.tac_model(ts, premises)

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {tactics}")

        return tactics


    def _get_tactics(self, goal):
        search_node = goal
        ts = search_node.goal

        # Get full set of suggestions for goal if it hasn't been computed already
        if ts not in self.remaining_tacs:
            logger.info(f'Generating tacs for {ts}')
            tacs = self._generate_tactics(ts)
            tacs.reverse()
            self.remaining_tacs[ts] = tacs

        suggestions = self.remaining_tacs[ts]

        # if we've exhausted all options
        if not suggestions:
            search_node.is_explored = True
            return None

        tactic, logprob = suggestions.pop()

        logger.info(f'Running {tactic}, node visits: {len(suggestions)}')

        return tactic, logprob



# DPO loss from paper,

# import torch.nn.functional as F
# def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
#     """
#     pi_logps: policy logprobs, shape (B,)
#     ref_logps: reference model logprobs, shape (B,)
#     yw_idxs: preferred completion indices in [0, B-1], shape (T,)
#     yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
#     beta: temperature controlling strength of KL penalty
#     Each pair of (yw_idxs[i], yl_idxs[i]) represents the
#     indices of a single preference pair.
#     """
#     pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
#     ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
#     pi_logratios = pi_yw_logps - pi_yl_logps
#     ref_logratios = ref_yw_logps - ref_yl_logps
#     losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
#     rewards = beta * (pi_logps - ref_logps).detach()
#     return losses, rewards
