"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pickle
import sys
import time
import traceback
from typing import Any

import hydra
import ray
import torch
import wandb
from lean_dojo import (
    Pos,
    Theorem,
    LeanGitRepo,
)
from loguru import logger
from omegaconf import OmegaConf
from ray.util.actor_pool import ActorPool

# from experiments.holist_supervised import config_to_dict
from experiments.reprover.common import set_logger
from experiments.reprover.common import zip_strict
from experiments.reprover.generator.model import RetrievalAugmentedGenerator
from experiments.reprover.goal_model_step.model import StepGoalModel
from refactor.get_lean_theorems import _get_theorems
from refactor.leandojo_env import LeanDojoEnv, EnvInitError
from refactor.proof_node import *
from refactor.search_result import SearchResult


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


# todo add tac/search model train functions, which can be lightning data modules
@ray.remote(num_gpus=0.01)
class EndToEndProver:
    def __init__(self, timeout, search_model, tac_model, directory):
        self.timeout = timeout
        self.search_model = search_model
        self.tac_model = tac_model

        self.total_time = 0
        self.search_time = 0
        self.tac_time = 0
        self.env_time = 0
        self.num_expansions = 0

        self.tac_trace = []

        self.dir = directory
        self.remaining_tacs = {}

    def _process_trace(self, theorem):
        root = self.search_model.get_root()
        nodes = self.search_model.get_nodes()
        traces = self.search_model.get_traces()

        if root.status == Status.PROVED:
            proof = [e.tactic for e in root.extract_proof()]
        else:
            proof = None

        result = SearchResult(
            theorem=f'{theorem}',
            status=root.status,
            proof=proof,
            tree=root,
            nodes=nodes,
            total_time=self.total_time,
            tac_time=self.tac_time,
            search_time=self.search_time,
            env_time=self.env_time,
            num_expansions=self.num_expansions,
            tac_trace=self.tac_trace,
            search_trace=traces,
            num_nodes=len(nodes)
        )

        logger.info(f'Result: {result}')

        with open(f"{self.dir}/{theorem}", "wb") as f:
            pickle.dump(result, f)

        return result

    # todo change based on fixed expansions vs one at a time
    def get_tactics(self, goals, premises):
        search_node = goals
        ts = search_node.goal

        # Get full set of suggestions for goal if it hasn't been computed already
        if ts not in self.remaining_tacs:
            logger.debug(f'Generating tacs for {ts}')
            tacs = ray.get(self.tac_model.get_tactics.remote(ts, premises))
            tacs.reverse()
            self.remaining_tacs[ts] = tacs

        suggestions = self.remaining_tacs[ts]

        # if we've exhausted all options
        if not suggestions:
            search_node.is_explored = True
            return None

        tactic, logprob = suggestions.pop()

        logger.debug(f'Running {tactic}, node visits: {len(suggestions)}')

        return tactic, logprob

    # todo treat goals and tactics as a list, and loop over env.run based on the # goals and tactics
    def _step(self, env):
        t0 = time.monotonic()
        goals = self.search_model.get_goals()
        self.search_time += time.monotonic() - t0

        t0 = time.monotonic()

        if not goals:
            raise Exception("No valid goals")

        premises = env.retrieve_premises()

        tactics = self.get_tactics(goals, premises)

        if not tactics:
            return

        self.tac_time += time.monotonic() - t0

        t0 = time.monotonic()
        response = env.run_tactic(goals, tactics)
        self.env_time += time.monotonic() - t0

        # record tactic and response in trace
        self.tac_trace.append(response)

        self.num_expansions += 1

        t0 = time.monotonic()
        self.search_model.process_response(response)
        self.search_time += time.monotonic() - t0

    def search(self, env):
        with torch.no_grad():
            try:
                self._search(env)
            except Exception as e:
                # will only be raised if there is no root from search (e.g. error loading environment)
                logger.warning(f"Error in search {e}")
                traceback.print_exc()
                self.search_model.__init__(self.search_model.goal_model)
                self.search_model.root = ErrorNode(EnvironmentError(str(e)))

        result = self._process_trace(env.thm.full_name)

        return result

    def _search(self, env) -> None:
        try:
            root = None
            self.search_time = 0
            self.tac_time = 0
            self.env_time = 0
            self.num_expansions = 0
            self.tac_trace = []

            with env as (env, root):
                time_start = time.monotonic()
                self.search_model.reset(root)
                logger.info(f'Attempting to prove {root}')

                while True:
                    try:
                        self._step(env)
                    except Exception as e:
                        # todo make env timeout error
                        if not (self.env_time >= self.timeout):
                            logger.warning(f"Exception not timeout: {e}")
                            traceback.print_exc()
                            root.status = Status.FAILED

                    self.total_time = time.monotonic() - time_start

                    # timeout only on environment, since model calls are queued and blocking
                    if self.env_time >= self.timeout:
                        if root.status == Status.PROVED:
                            logger.info("Found a proof but timed out.")
                        root.status = Status.OPEN
                        logger.info("Search timed out.")
                        break

                    if root.status == Status.FAILED:
                        logger.info("Failed early!")
                        break

                    if root.status == Status.PROVED:
                        logger.info("Found a proof!")
                        break
        except Exception as e:
            logger.warning(f'Environment error {e}')
            traceback.print_exc()
            if root:
                root.status = Status.FAILED
            else:
                raise Exception


class Search:
    @abstractmethod
    def reset(self, root):
        return

    @abstractmethod
    def get_goals(self):
        return

    @abstractmethod
    def process_response(self, response):
        return


@ray.remote(num_cpus=0.5, num_gpus=0.225)
class GoalModel:
    def __init__(self, model):
        self.model = model

    def run(self, goals):
        scores = self.model.batch_generate(goals)
        return scores


class UpDown(Search):
    def __init__(self, goal_model):
        self.goal_model = goal_model
        self.remaining_tacs = {}
        self.search_trace = []
        self.nodes = {}
        self.root = None

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root
        self.nodes[root.goal] = root

        # initialise scores for root
        node_goals = ['<extra_id_0>' + self.root.goal]
        # scores = self.goal_model.batch_generate(node_goals)
        scores = ray.get(self.goal_model.run.remote(node_goals))

        self.root.provable_score = scores[0]
        self.root.up_score = scores[0]

    # todo sample?
    # todo return goal plus the context (i.e. best fringe)?
    def get_goals(self):
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

        if best_node:
            self.search_trace.append((best_node.goal, best_score))
        else:
            self.search_trace.append((None, -math.inf))

        return best_node

    def _up_step(self, node):
        if node.out_edges:
            best_score = -math.inf
            for edge in node.out_edges:
                edge_score = 0
                for sib in edge.dst:
                    edge_score += sib.up_score

                if edge_score >= best_score:
                    best_score = edge_score

            if node.visit_count >= node.max_expansions:
                node.provable_score = -math.inf
                node.is_explored = True

            up_score = max(node.provable_score, best_score)

            # todo scale breadth as it's explored
            if up_score != node.up_score:
                node.up_score = up_score
                parents = set([edge.src for edge in node.in_edges])
                for parent in parents:
                    self._up_step(parent)

    # Assume response is a single edge in this case
    def process_response(self, response):
        search_node = response.src
        result = response.dst

        # find new nodes from response, and compute their provable score
        new_nodes = []
        for result_node in result:
            if isinstance(result_node, InternalNode):
                if result_node.goal not in self.nodes:
                    new_nodes.append(result_node)

        if new_nodes:
            # compute provable_score/up_score for new internal nodes
            node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]

            scores = ray.get(self.goal_model.run.remote(node_goals))

            for i, node_ in enumerate(new_nodes):
                node_.provable_score = (scores[i] + (node_.depth * math.log(0.95))).item()
                node_.up_score = node_.provable_score

        for result_node in result:
            # Record the new node and add it to the search queue.
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(response)
                self.nodes[result_node.goal] = result_node

        if search_node.out_edges:
            search_node.out_edges = search_node.out_edges + [response]
        else:
            search_node.out_edges = [response]

        self._up_step(search_node)

        return

    def get_root(self):
        return self.root

    def get_traces(self):
        return self.search_trace

    def get_nodes(self):
        return self.nodes


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, env):
        return


# todo make tac_gen and retriever more system agnostic
@ray.remote(num_cpus=0.5, num_gpus=0.225)
class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, ts, premises):
        path, theorem, position = premises

        tactics = self.tac_model.generate(
            state=ts,
            file_path=path,
            theorem_full_name=theorem.full_name,
            theorem_pos=position,
            num_samples=self.num_sampled_tactics,
        )

        logger.debug(f"Tactic suggestions: {tactics}")
        return tactics


# # set to 1 / (gpu_mem //  required_mem)
# @ray.remote(num_gpus=1)
# class GpuProver(EndToEndProver):
#     """Ray actor for running an instance of `EndToEndProver` on a GPU."""
#
#     def __init__(
#             self,
#             ckpt_path: str,
#             goal_path: str,
#             indexed_corpus_path: Optional[str],
#             timeout: int,
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
#         tac_model = ReProverTacGen(tac_model=tac_gen)
#
#         goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)
#
#         search_model = UpDown(goal_model)
#
#         super().__init__(
#             timeout=timeout,
#             search_model=search_model,
#             tac_model=tac_model
#         )
#
#
# @ray.remote
# class CpuProver(EndToEndProver):
#     """Ray actor for running an instance of `EndToEndProver` on a GPU."""
#
#     def __init__(
#             self,
#             ckpt_path: str,
#             goal_path: str,
#             indexed_corpus_path: Optional[str],
#             timeout: int,
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
#         search_model = UpDown(goal_model)
#
#         tac_model = ReProverTacGen(tac_model=tac_gen)
#
#         super().__init__(
#             timeout=timeout,
#             search_model=search_model,
#             tac_model=tac_model
#         )
#

class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
            self,
            ckpt_path: str,
            goal_path: str,
            indexed_corpus_path: Optional[str],
            num_cpus: int,
            with_gpus: bool,
            timeout: int,
            log_dir: str
    ) -> None:

        # self.distributed = num_cpus > 1
        self.distributed = False

        if not self.distributed:
            with_gpus = True
            self.num_gpus = 2

            ray.init(num_gpus=self.num_gpus)

            # todo make multiple copies for tac_gen and goal_model depending on the number of GPUs and available GPU VRAM

            device = torch.device("cuda") if with_gpus else torch.device("cpu")

            prover_pool = []

            # todo test performance loading these before and passing object reference
            for _ in range(4):
                tac_gen = RetrievalAugmentedGenerator.load(
                    ckpt_path, device=device, freeze=True
                )
                if tac_gen.retriever is not None:
                    assert indexed_corpus_path is not None
                    tac_gen.retriever.load_corpus(indexed_corpus_path)

                goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)

                tac_model = ReProverTacGen.remote(tac_model=tac_gen)

                goal_model = GoalModel.remote(goal_model)

                search_model = UpDown(goal_model)

                prover_pool.extend([EndToEndProver.remote(
                    tac_model=tac_model, search_model=search_model, timeout=timeout, directory=log_dir
                    # ) for _ in range(num_cpus // self.num_gpus)])
                ) for _ in range(1)])

            self.prover_pool = ActorPool(prover_pool)

            return

        if with_gpus:
            logger.info(f"Launching {num_cpus} GPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=num_cpus)
            # provers = [
            #     GpuProver.remote(
            #         ckpt_path,
            #         goal_path,
            #         indexed_corpus_path,
            #         timeout=timeout,
            #     )
            #     for _ in range(num_cpus)
            # ]
        else:
            logger.info(f"Launching {num_cpus} CPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=0)
            # provers = [
            #     CpuProver.remote(
            #         ckpt_path,
            #         goal_path,
            #         indexed_corpus_path,
            #         timeout=timeout
            #     )
            #     for _ in range(num_cpus)
            # ]

        # self.prover_pool = ActorPool(provers)

    def search_unordered(
            self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> Any:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(LeanDojoEnv(repo, x[0], x[1], 6000)),
                    zip_strict(theorems, positions),
                )
            )

            # self.prover.search(repo, thm, pos)
            # self.prover.search(LeanDojoEnv(repo, thm, pos, 600))
            # for thm, pos in zip_strict(theorems, positions)

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(LeanDojoEnv(repo, x[0], x[1], 6000)),
                    zip_strict(theorems, positions),
                )
            )

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results


@hydra.main(
    config_path="/home/sean/Documents/bait/experiments/configs/experiments")  # , config_name="experiments/holist_eval")
def main(config) -> None:
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)
    os.makedirs(config.exp_config.directory + '/traces', exist_ok=True)

    if config.exp_config.resume:
        # wandb.init(project=config.logging_config.project,
        #            name=config.exp_config.name,
        #            config=config_to_dict(config),
        #            dir=config.exp_config.directory,
        #            resume='must',
        #            id=config.logging_config.id,
        #            mode='offline' if config.logging_config.offline else 'online'
        #            )

        prev_theorems = glob.glob(config.exp_config.directory + '/traces/*')
    # else:
    #     wandb.init(project=config.logging_config.project,
    #                name=config.exp_config.name,
    #                config=config_to_dict(config),
    #                dir=config.exp_config.directory,
    #                mode='offline' if config.logging_config.offline else 'online'
    #                )

    set_logger(config.verbose)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config: {config}")

    repo, theorems, positions = _get_theorems(config)

    # Remove proven theorems if resuming
    if config.exp_config.resume:

        final_theorems = []
        final_positions = []

        for i, theorem in enumerate(theorems):
            if theorem.full_name in prev_theorems:
                continue
            elif theorem.full_name == 'nat.arithmetic_function.zeta_mul_pow_eq_sigma':
                final_theorems.append(theorem)
                final_positions.append(positions[i])
            else:
                # final_theorems.append(theorem)
                # final_positions.append(positions[i])
                continue

        theorems = final_theorems
        positions = final_positions

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        config.ckpt_path,
        config.goal_path,
        config.indexed_corpus_path,
        num_cpus=config.num_cpus,
        with_gpus=config.with_gpus,
        timeout=config.timeout,
        log_dir=config.exp_config.directory + '/traces'
    )

    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )
    logger.info(f"Pass@1: {num_proved / (num_proved + num_failed)}")

    # Save the results.
    if config.exp_id is not None:
        pickle_path = f"{config.exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")


if __name__ == '__main__':
    main()

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
