from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pickle
import random
import sys
import time
import traceback
from subprocess import CalledProcessError

from lean_dojo.utils import execute

import hydra
import ray
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

from data.holist.utils import io_util
from environments.holist import proof_assistant_pb2
from refactor.common import set_logger
from refactor.common import zip_strict
from refactor.get_lean_theorems import _get_theorems
from refactor.holist_env import HOListEnv
from refactor.leandojo_env import LeanDojoEnv
from refactor.process_traces import get_traces
from refactor.proof_node import *
from refactor.search_models import get_search_model
from refactor.search_result import SearchResult
from refactor.tac_models import get_tac_model


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def get_thm_name(env, thm):
    if env == 'holist':
        return thm.fingerprint
    elif env == 'leandojo':
        return thm.full_name
    else:
        raise NotImplementedError


class EndToEndProver:
    def __init__(self, timeout, search_model, tac_model, directory, env_name='leandojo', iteration=0):
        self.timeout = timeout
        self.search_model = search_model
        self.tac_model = tac_model
        self.env_name = env_name

        self.total_time = 0
        self.search_time = 0
        self.tac_time = 0
        self.env_time = 0
        self.num_expansions = 0

        self.trace = []

        self.dir = f'{directory}/traces/{iteration}'
        self.error_dir = f'{directory}/{iteration}/error_logs'

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

        # maps goals to tactics once generated
        self.remaining_tacs = {}

    def _process_trace(self, theorem):
        root = self.search_model.root
        nodes = self.search_model.nodes

        if root.status == Status.PROVED:
            proof = [e.tactic for e in root.extract_proof()]
        else:
            proof = None

        result = SearchResult(
            theorem=theorem,
            status=root.status,
            proof=proof,
            tree=root,
            nodes=nodes,
            total_time=self.total_time,
            tac_time=self.tac_time,
            search_time=self.search_time,
            env_time=self.env_time,
            num_expansions=self.num_expansions,
            trace=self.trace,
            num_nodes=len(nodes),
            data={'search_trace': self.search_model.search_trace} if hasattr(self.search_model, 'search_trace') else {}
        )

        with open(f"{self.dir}/{get_thm_name(self.env_name, theorem)}", "wb") as f:
            pickle.dump(result, f)

        return

    def get_tactics(self, goals, premises, tacs_per_goal=64):
        suggestions = []
        for search_node, prob in goals:
            assert not search_node.is_explored
            ts = search_node.goal

            # Get full set of suggestions for goal if it hasn't been computed already
            if ts not in self.remaining_tacs:
                tacs = ray.get(self.tac_model.get_tactics.remote(ts, premises))
                tacs.reverse()
                self.remaining_tacs[ts] = tacs

            remaining_tacs = self.remaining_tacs[ts]

            for _ in range(tacs_per_goal):
                if remaining_tacs:
                    tactic, logprob = remaining_tacs.pop()
                    suggestions.append(((search_node, prob), (tactic, logprob)))
                else:
                    search_node.is_explored = True
                    continue

        return suggestions

    def _step(self, env):
        t0 = time.monotonic()
        goals = self.search_model.get_goals()
        self.search_time += time.monotonic() - t0

        t0 = time.monotonic()

        if not goals:
            raise Exception("No valid goals")

        premises = env.premises

        suggestions = self.get_tactics(goals, premises)

        if not suggestions:
            return

        self.tac_time += time.monotonic() - t0

        responses = []
        for goal, tactic in suggestions:
            t0 = time.monotonic()
            logger.debug(f'Running {tactic}, goal: {goal}')
            response = env.run_tactic(goal, tactic)
            self.env_time += time.monotonic() - t0

            self.trace.append(response)
            responses.append(response)
            self.num_expansions += 1

        t0 = time.monotonic()
        self.search_model.process_responses(responses)
        self.search_time += time.monotonic() - t0

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
                self.log_error(str(e), get_thm_name(self.env_name, env.thm))

                root = ErrorNode(EnvironmentError(str(e)))
                self.search_model.reset(root)

        self._process_trace(env.thm)

        return self.search_model.root.status == Status.PROVED

    def _search(self, env) -> None:
        try:
            root = None
            self.search_time = 0
            self.tac_time = 0
            self.env_time = 0
            self.num_expansions = 0
            self.trace = []

            with env as (env, root):
                time_start = time.monotonic()
                self.search_model.reset(root)
                logger.info(f'Attempting to prove {root}')

                while True:
                    try:
                        self._step(env)
                    except Exception as e:
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


def get_env(cfg):
    if cfg == 'leandojo':
        return LeanDojoEnv
    elif cfg == 'holist':
        return HOListEnv
    else:
        raise NotImplementedError


class DistributedProver:
    """
    A distributed prover that uses Ray to parallelize the proof search.
    """

    def __init__(self, config, iteration=0) -> None:
        self.total_timeout = config.total_timeout

        self.iteration = iteration

        ray.init(num_gpus=config.num_gpus, num_cpus=config.num_cpus)

        device = torch.device("cuda") if config.with_gpus else torch.device("cpu")

        prover_pool = []

        for i in range(config.logical_gpus):
            tac_model = get_tac_model(config.tac_model, device)
            search_model = get_search_model(config.search_model, device)

            prover_pool.extend(
                [ray.remote(num_gpus=config.gpu_per_prover, num_cpus=config.cpu_per_prover)(EndToEndProver).remote(
                    tac_model=tac_model, search_model=search_model, timeout=config.env_timeout,
                    directory=config.exp_config.directory, env_name=config.env, iteration=iteration
                ) for _ in range(config.provers_per_gpu)])

        self.prover_pool = ActorPool(prover_pool)

        return

    def search_unordered(self, theorems, resume_proven=0, resume_step=0, env='leandojo'):
        try:
            iteration = self.iteration
            env_func = get_env(env)
            results_ = self.prover_pool.map_unordered(
                lambda p, thm: p.search.remote(env_func(thm, self.total_timeout)),
                theorems,
            )

            proven = resume_proven
            for i, res in enumerate(results_):
                if res:
                    proven += 1
                wandb.log({'Step': i + 1 + resume_step, 'Proven': proven, 'Iteration': iteration})

            return proven

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)


def get_holist_theorems(thm_db, prev_theorems):
    theorem_db = io_util.load_theorem_database_from_file(
        str(thm_db))

    # todo filter by config split, library etc.
    theorems = [thm for thm in theorem_db.theorems if thm.tag == proof_assistant_pb2.Theorem.THEOREM]

    # Remove proven theorems if resuming
    final_theorems = []

    for i, theorem in enumerate(theorems):
        if theorem.fingerprint in prev_theorems:
            continue
        else:
            final_theorems.append(theorem)

    theorems = final_theorems
    theorems = list(zip_strict(theorems, [theorem_db] * len(theorems)))

    return theorems


def get_lean_thms(config, prev_theorems):
    repo, theorems, positions = _get_theorems(config)

    # Remove proven theorems if resuming
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

    theorems = list(zip_strict([repo] * len(theorems), theorems, positions))

    return theorems


def get_theorems(cfg, prev_theorems):
    if cfg.env == 'leandojo':
        return get_lean_thms(cfg, prev_theorems)
    elif cfg.env == 'holist':
        return get_holist_theorems(cfg.env_config.path_theorem_database, prev_theorems)


@hydra.main(config_path="../experiments/configs/experiments")
def main(config) -> None:
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    prev_theorems = []
    prev_proven = 0
    iteration = 0

    if config.exp_config.resume:
        iteration = config.resume_iteration
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   resume='must',
                   id=config.logging_config.id,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

        # prev_theorems = get_traces(f'{config.exp_config.directory}/traces/{iteration}/*')
        trace_dir = glob.glob(f'{config.exp_config.directory}/traces/{iteration}/*')

        logger.info('Loading previous proofs..')

        for file in tqdm(trace_dir):
            with open(file, "rb") as f:
                trace = pickle.load(f)
            if trace.proof:
                prev_proven += 1
            prev_theorems.append(get_thm_name(config.env, trace.theorem))

        logger.info(f'Resuming from {prev_proven} proofs over {len(prev_theorems)} attempts..')
    else:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

    theorems = get_theorems(config, prev_theorems)

    set_logger(config.log_level)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config: {config}")

    if config.shuffle:
        random.shuffle(theorems)

    theorems = theorems[:config.num_theorems]

    num_iterations = config.num_iterations if hasattr(config, 'num_iterations') else 1

    for i in range(num_iterations):
        prover = DistributedProver(config, iteration)

        logger.info(f'Attempting {len(theorems)} proofs..')

        num_proven = prover.search_unordered(theorems, resume_step=len(prev_theorems),
                                             resume_proven=prev_proven, env=config.env)

        # log as error for now, to minimise output for parent processes
        logger.error(f"Pass@1: {num_proven / config.num_theorems}")

        # todo reload checkpoints for newly trained models
        if hasattr(config, 'train_after_eval') and num_iterations > 1:
            for cmd in config.train_after_eval:
                logger.info(f'Running training with {cmd}')

                try:
                    _, err = execute(cmd, capture_output=True)
                except CalledProcessError as ex:
                    logger.error(ex)
                    logger.error("Failed to train.")


if __name__ == '__main__':
    main()
