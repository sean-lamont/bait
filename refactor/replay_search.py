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

from experiments.reprover.common import set_logger
from experiments.reprover.common import zip_strict
from experiments.reprover.generator.model import RetrievalAugmentedGenerator
from experiments.reprover.goal_model_step.model import StepGoalModel
from refactor.get_lean_theorems import _get_theorems
from refactor.leandojo_env import LeanDojoEnv
from refactor.proof_node import *
from refactor.search_models import UpDown, GoalModel, BestFS
from refactor.search_result import SearchResult
from refactor.tac_models import ReProverTacGen


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


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

        self.dir = directory + '/traces'
        self.error_dir = directory + '/error_logs'
        # maps goals to tactics once generated
        self.remaining_tacs = {}

    def _process_trace(self, theorem):
        root = self.search_model.root
        nodes = self.search_model.nodes
        traces = self.search_model.search_trace

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

        with open(f"{self.dir}/{theorem}", "wb") as f:
            pickle.dump(result, f)

        return result

    def get_tactics(self, goals, premises, tacs_per_goal=64):
        suggestions = []
        for search_node in goals:
            assert not search_node.is_explored
            ts = search_node.goal

            # Get full set of suggestions for goal if it hasn't been computed already
            if ts not in self.remaining_tacs:
                logger.debug(f'Generating tacs for {ts}')
                tacs = ray.get(self.tac_model.get_tactics.remote(ts, premises))
                if len(tacs) < tacs_per_goal:
                    logger.debug(f'Fewer than max tactics generated for {search_node.goal}')
                tacs.reverse()
                self.remaining_tacs[ts] = tacs

            remaining_tacs = self.remaining_tacs[ts]

            for _ in range(tacs_per_goal):
                if remaining_tacs:
                    tactic, logprob = remaining_tacs.pop()
                    suggestions.append((search_node, (tactic, logprob)))
                else:
                    search_node.is_explored = True
                    continue

        return suggestions

    def _step(self, env, goal, tactic):
        goal = goal

        suggestions = tactic

        if not suggestions:
            return

        logger.debug(f'Running {tactic}, goal: {goal}')
        response = env.run_tactic(goal, tactic)

        self.search_model.process_response(response)

    def log_error(self, msg, theorem):
        with open(f"{self.error_dir}/{theorem}", "a") as f:
            f.writelines([msg])

    def search(self, env):
        with torch.no_grad():
            try:
                self._search(env)
            except Exception as e:
                logger.warning(f'Environment error {e}')
                # will only be raised if there is no valid root from search (e.g. error loading environment)
                # self.search_model.__init__(self.search_model.goal_model)
                self.search_model.__init__()
                self.search_model.root = ErrorNode(EnvironmentError(str(e)))
                self.log_error(str(e), env.thm.full_name)

        result = self._process_trace(env.thm.full_name)

        return result

    def replay(self, env, trace) -> None:
        try:
            root = trace.tree
            self.search_time = 0
            self.tac_time = 0
            self.env_time = 0
            self.num_expansions = 0
            tac_trace = trace.tac_trace
            search_trace = trace.search_trace

            with env as (env, root):
                time_start = time.monotonic()
                self.search_model.reset(root)
                logger.info(f'Attempting to prove {root}')

                for i in range(len(tac_trace)):
                    try:
                        self._step(env, search_trace[i], tac_trace[i])
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
            if root:
                logger.warning(f"Error in search {e}")
                root.status = Status.FAILED
                self.log_error(str(e))
            else:
                raise Exception(e)


class DistributedProver:
    """
    A distributed prover that uses Ray to parallelize the proof search.
    """

    # todo more agnostic, loaded from config
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
            self.num_gpus = 1
            self.cpus_per_gpu = 8

            ray.init(num_gpus=1)

            device = torch.device("cuda") if with_gpus else torch.device("cpu")

            prover_pool = []

            for _ in range(self.num_gpus):
                # todo test performance loading these before and passing object reference
                tac_gen = RetrievalAugmentedGenerator.load(
                    ckpt_path, device=device, freeze=True
                )
                if tac_gen.retriever is not None:
                    assert indexed_corpus_path is not None
                    tac_gen.retriever.load_corpus(indexed_corpus_path)

                # goal_model = StepGoalModel.load(goal_path, device=device, freeze=True)

                # todo best way to parameterise resource allocation
                tac_model = ray.remote(num_gpus=0.45)(ReProverTacGen).remote(tac_model=tac_gen)

                # goal_model = ray.remote(num_gpus=0.225, num_cpus=0.5)(GoalModel).remote(goal_model)

                # search_model = UpDown(goal_model)

                search_model = BestFS()

                prover_pool.extend([ray.remote(num_gpus=0.01)(EndToEndProver).remote(
                    tac_model=tac_model, search_model=search_model, timeout=timeout, directory=log_dir
                ) for _ in range(self.cpus_per_gpu)])

            self.prover_pool = ActorPool(prover_pool)

            return

    # todo get env / theorems from config (e.g. HOListEnv, then theorems/positions and arguments to env will be
    #  different)

    def search_unordered(
            self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos], iteration=0
    ) -> Any:

        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the
        input."""

        try:
            results = self.prover_pool.map_unordered(
                lambda p, x: p.search.remote(LeanDojoEnv(repo, x[0], x[1], 6000)),
                zip_strict(theorems, positions),
            )

            proven = 0
            for i, res in enumerate(results):
                logger.info(f'Result: {res}')
                if res.proof:
                    proven += 1
                    wandb.log({'Step': i + 1, 'Proven': proven, 'Iteration': iteration})

            return results

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)


@hydra.main(config_path="../experiments/configs/experiments")
def main(config) -> None:
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)
    os.makedirs(config.exp_config.directory + '/traces', exist_ok=True)
    os.makedirs(config.exp_config.directory + '/error_logs', exist_ok=True)

    # todo make this system agnostic
    repo, theorems, positions = _get_theorems(config)

    if config.exp_config.resume:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   resume='must',
                   id=config.logging_config.id,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

        # Remove proven theorems if resuming
        prev_theorems = glob.glob(config.exp_config.directory + '/traces/*')

        final_theorems = []
        final_positions = []

        for i, theorem in enumerate(theorems):
            if theorem.full_name in prev_theorems:
                continue
            else:
                final_theorems.append(theorem)
                final_positions.append(positions[i])

        theorems = final_theorems
        positions = final_positions

    else:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

    set_logger(config.verbose)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config: {config}")

    # todo loop over iterations
    # for i in range(len(self.num_iterations)): ...
    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        config.ckpt_path,
        config.goal_path,
        config.indexed_corpus_path,
        num_cpus=config.num_cpus,
        with_gpus=config.with_gpus,
        timeout=config.timeout,
        log_dir=config.exp_config.directory
    )

    results = prover.search_unordered(repo, theorems, positions, iteration=0)

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
        f"Iteration done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )
    logger.info(f"Pass@1: {num_proved / (num_proved + num_failed)}")

    # todo add end-to-end training
    # todo add tac/search model train functions, which can be lightning data modules
    # self.process_result(results, config) (e.g. process and add results to mongodb)
    # if self.tac_train:
    # self.tac_model.train (take recently updated database, and retrain tac_model)
    # also would update retriever e.g. refresh embedding generation
    # if self.goal.train:
    # self.goal_model.train (likewise for goal_model)


if __name__ == '__main__':
    main()