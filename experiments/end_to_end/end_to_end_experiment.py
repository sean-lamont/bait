from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pickle
import random
import re
import sys
import time
import traceback
from subprocess import CalledProcessError

# for Lean 3
os.environ['CONTAINER'] = 'docker'

import hydra
import ray
import torch
import wandb
from lean_dojo.utils import execute
from loguru import logger
from omegaconf import OmegaConf
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

from experiments.end_to_end.common import set_logger
from experiments.end_to_end.env_helper import get_thm_name, get_env, get_theorems
from experiments.end_to_end.proof_node import *
from experiments.end_to_end.search_result import SearchResult
from models.end_to_end.search_models.get_search_model import get_search_model
from models.end_to_end.tactic_models.tac_models import get_tac_model
from utils.utils import config_to_dict


"""

Main script for running live proof search experiments.

"""


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

        data = {}

        if hasattr(self.search_model, 'search_trace'):
            data['search_trace'] = self.search_model.search_trace

        data['env'] = self.env_name

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
            data=data
        )

        try:
            with open(os.path.join(self.dir, get_thm_name(self.env_name, theorem)), "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Couldn't save trace for {theorem}, saving status only: \n Error: {e}")

            root = ErrorNode(EnvironmentError(str(e)))

            result = SearchResult(
                theorem=theorem,
                status=root.status,
                proof=proof,
                tree=root,
                nodes={},
                total_time=self.total_time,
                tac_time=self.tac_time,
                search_time=self.search_time,
                env_time=self.env_time,
                num_expansions=self.num_expansions,
                trace=self.trace,
                num_nodes=len(nodes),
                data={}
            )

            with open(os.path.join(self.dir, get_thm_name(self.env_name, theorem)), "wb") as f:
                pickle.dump(result, f)

        return 1 if proof else 0

    def get_tactics(self, goals, premises, tacs_per_goal=64):
        suggestions = []
        for search_node, prob in goals:
            assert not search_node.is_explored
            ts = search_node.goal

            # Get full set of suggestions for goal if it hasn't been computed already
            if ts not in self.remaining_tacs:
                tacs = self.tac_model.get_tactics(search_node, premises)
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
        with open(os.path.join(self.error_dir, theorem), "a") as f:
            f.writelines([msg])

    def search(self, env):
        with torch.no_grad():
            try:
                self._search(env)
            except Exception as e:
                # will only be raised from _search if there is no valid root from search, e.g. error loading environment
                logger.warning(f'Environment initialisation error {e}')
                try:
                    err_name = get_thm_name(self.env_name, env.thm)
                    err_file = os.path.join(self.error_dir, err_name)
                    traceback.print_exc(file=open(err_file, 'a'))
                except:
                    logger.warning(f"Couldn't log error for {err_name}:")
                    traceback.print_exc()

                # log the trace for this attempt as a single error node
                root = ErrorNode(EnvironmentError(str(e)))
                self.search_model.reset(root)

        try:
            proven = self._process_trace(env.thm)
        except:
            proven = 0
            logger.warning(f"Error processing trace for {env.thm}")
            err_name = get_thm_name(self.env_name, env.thm)
            try:
                err_file = os.path.join(self.error_dir, err_name)
                traceback.print_exc(file=open(err_file, 'a'))
            except:
                logger.warning(f"Couldn't log error for {err_name}:")
                traceback.print_exc()

        return proven

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
                            logger.warning(f"Exception not timeout for {get_thm_name(self.env_name, env.thm)}: {e}")
                            traceback.print_exc()
                            root.status = Status.FAILED
                            env._cleanup()
                            raise Exception(e)

                    self.total_time = time.monotonic() - time_start

                    # two timeouts: env time, and env.timeout.
                    # env_time is maximum time in environment, env.timeout is total time including tactics
                    if self.env_time >= self.timeout or self.total_time >= env.timeout:
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
            else:
                raise Exception(f"Could not initialise root: {e}")


class DistributedProver:
    """
    A distributed prover that uses Ray to parallelize the proof search.
    """

    def __init__(self, config, iteration=0) -> None:
        self.total_timeout = config.total_timeout

        self.iteration = iteration

        ray.init(num_gpus=config.num_gpus, num_cpus=config.num_cpus)

        device = torch.device("cuda") if config.num_gpus > 0 else torch.device("cpu")

        prover_pool = []

        if config.num_gpus == 0:
            tac_model = get_tac_model(config.tac_model, device)
            search_model = get_search_model(config.search_model, device)

            prover_pool.append(ray.remote(num_cpus=config.cpu_per_prover)(EndToEndProver).remote(
                tac_model=tac_model, search_model=search_model, timeout=config.env_timeout,
                directory=config.exp_config.directory, env_name=config.env_config.env, iteration=iteration
            ))

        else:
            for i in range(config.logical_gpus):
                tac_model = get_tac_model(config.tac_model, device)
                search_model = get_search_model(config.search_model, device)

                prover_pool.extend(
                    [ray.remote(num_gpus=config.gpu_per_prover, num_cpus=config.cpu_per_prover)(EndToEndProver).remote(
                        tac_model=tac_model, search_model=search_model, timeout=config.env_timeout,
                        directory=config.exp_config.directory, env_name=config.env_config.env, iteration=iteration
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


@hydra.main(config_path="../../configs")
def main(config) -> None:
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    prev_theorems = []
    prev_proven = 0
    cur_iteration = 0

    if config.exp_config.resume:
        cur_iteration = config.resume_iteration
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   resume='must',
                   id=config.logging_config.id,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

        trace_dir = glob.glob(f'{config.exp_config.directory}/traces/{cur_iteration}/*')

        logger.info('Loading previous proofs..')

        for file in tqdm(trace_dir):
            with open(file, "rb") as f:
                try:
                    trace = pickle.load(f)
                except:
                    continue
            if trace.proof:
                prev_proven += 1
            prev_theorems.append(get_thm_name(config.env_config.env, trace.theorem))

        logger.info(f'Resuming from {prev_proven} proofs over {len(prev_theorems)} attempts..')
    else:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   mode='offline' if config.logging_config.offline else 'online'
                   )

    theorems = get_theorems(config.env_config, prev_theorems)

    set_logger(config.log_level)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    if config.shuffle:
        random.shuffle(theorems)

    theorems = theorems[:config.env_config.num_theorems]

    num_iterations = config.num_iterations if hasattr(config, 'num_iterations') else 1

    for iteration in range(cur_iteration, num_iterations):
        prover = DistributedProver(config, iteration)

        logger.info(f'Attempting {len(theorems)} proofs..')

        num_proven = prover.search_unordered(theorems, resume_step=len(prev_theorems),
                                             resume_proven=prev_proven, env=config.env_config.env)

        # log as error for now, to minimise output for parent processes
        logger.error(f"Pass@1: {num_proven / config.env_config.num_theorems}")

        wandb.log({'Pass@1': num_proven / config.env_config.num_theorems, 'Iteration': iteration})

        ray.shutdown()

        if hasattr(config, 'train_after_eval') and num_iterations > 1:
            new_ckpt_dirs = []
            for cmd in config.train_after_eval:
                logger.info(f'Running training with {cmd}')

                try:
                    _, err = execute(cmd, capture_output=True)
                except CalledProcessError as ex:
                    logger.error(ex)
                    logger.error("Failed to train.")
                    raise Exception("Failed to train")

                m = re.search(r"checkpoint_dir: (\S+)", err)
                assert m is not None, err

                logger.warning(str(m.group(1)))
                new_ckpt_dirs.append(str(m.group(1)).split('$')[0])

                logger.info('Done.')

            to_update = config.update_checkpoints

            # update checkpoint paths in config with newly trained versions
            logger.info('Updating checkpoints..')
            for i, ckpt in enumerate(new_ckpt_dirs):
                model, ckpt_dir = to_update[i]
                setattr(getattr(config, model), ckpt_dir, str(ckpt))


if __name__ == '__main__':
    main()
