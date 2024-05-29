from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import random
import sys
import time
import traceback

# for Lean 3
os.environ['CONTAINER'] = 'docker'

import hydra
import ray
import torch
from loguru import logger
from omegaconf import OmegaConf
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

from experiments.end_to_end.common import set_logger, format_tactic
from experiments.end_to_end.env_helper import get_thm_name, get_env, get_theorems
from experiments.end_to_end.proof_node import *
from experiments.end_to_end.search_result import SearchResult
from models.end_to_end.tactic_models.tac_models import get_tac_model

"""

Replay human proofs, and generate additional data for each node in the proof path. 
Useful to get some 'negative' data to compare with the positive data from the human proofs.

"""


class ReplayProver:
    def __init__(self, timeout, tac_model, directory, env_name='leandojo', iteration=0):
        self.timeout = timeout
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

        self.nodes = {}
        self.root = None

    def _process_trace(self, theorem):
        root = self.root
        nodes = self.nodes

        if not root:
            return

        if root.status == Status.PROVED:
            proof = [e.tactic for e in root.extract_proof()]
        else:
            proof = None

        data = {'env': self.env_name}

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

        if proof:
            with open(os.path.join(self.dir, get_thm_name(self.env_name, theorem)), "wb") as f:
                pickle.dump(result, f)

        return

    def get_tactics(self, goals, premises, tacs_per_goal=63):
        suggestions = []
        for search_node in goals:
            ts = search_node.goal

            # Get full set of suggestions for goal if it hasn't been computed already
            if ts not in self.remaining_tacs:
                # tacs = ray.get(self.tac_model.get_tactics.remote(search_node, premises))
                tacs = self.tac_model.get_tactics(search_node, premises)
                tacs.reverse()
                self.remaining_tacs[ts] = tacs

            remaining_tacs = self.remaining_tacs[ts]

            for _ in range(tacs_per_goal):
                if remaining_tacs:
                    tactic, logprob = remaining_tacs.pop()
                    suggestions.append(((search_node, 1.0), (tactic, logprob)))
                else:
                    search_node.is_explored = True
                    continue

        return suggestions

    def _step(self, env):
        goals = self.proof_path

        t0 = time.monotonic()

        if not goals:
            raise Exception("No valid goals")

        premises = env.premises

        suggestions = self.get_tactics(goals, premises)

        if not suggestions:
            return

        self.tac_time += time.monotonic() - t0

        for goal, tactic in suggestions:
            t0 = time.monotonic()
            logger.debug(f'Running {tactic}, goal: {goal}')
            response = env.run_tactic(goal, tactic)
            self.env_time += time.monotonic() - t0

            self.trace.append(response)
            self.num_expansions += 1

            for result_node in response.dst:
                if isinstance(result_node, InternalNode):
                    if result_node.goal not in self.nodes:
                        self.nodes[result_node.goal] = result_node

        # run again for new nodes to get retrieved states, and to verify they aren't proven
        new_goals = []
        for node in self.nodes.values():
            if not node.is_explored:
                new_goals.append(node)

        suggestions = self.get_tactics(new_goals, premises)

        if not suggestions:
            return

        self.tac_time += time.monotonic() - t0

        for goal, tactic in suggestions:
            t0 = time.monotonic()
            logger.debug(f'Running {tactic}, goal: {goal}')
            response = env.run_tactic(goal, tactic)
            self.env_time += time.monotonic() - t0

            self.trace.append(response)
            self.num_expansions += 1

    def log_error(self, msg, theorem):
        with open(os.path.join(self.error_dir, theorem), "a") as f:
            f.writelines([msg])

    def search(self, env, proof):
        with torch.no_grad():
            try:
                ret = self._search(env, proof)
            except Exception as e:
                logger.warning(f'Environment error {e}')
                traceback.print_exc()
                # will only be raised if there is no valid root from search (e.g. error loading environment)
                self.log_error(str(e), get_thm_name(self.env_name, env.thm))
                ret = False

        self._process_trace(env.thm)

        return ret

    def _search(self, env, proof):
        try:
            root = None
            self.root = None
            self.search_time = 0
            self.tac_time = 0
            self.env_time = 0
            self.num_expansions = 0
            self.trace = []
            self.nodes = {}

            with env as (env, root):
                try:
                    time_start = time.monotonic()
                    logger.info(f'Replaying proof of {root}')

                    ordered_states = [root]
                    all_states = [root]

                    self.root = root

                    if not root:
                        raise Exception('Invalid root')

                    print('Proof:', proof)

                    for tactic in proof:
                        state = ordered_states.pop()
                        response = env.run_tactic((state, 1.0), (tactic, 1.0))

                        # logger.info(f"Running {tactic} on {state}, got {response}")

                        if isinstance(response.dst[0], InternalNode):
                            ordered_states.extend(reversed(response.dst))
                        elif isinstance(response.dst[0], ErrorNode):
                            logger.info(f'Error in proof: {response.dst[0]}')
                            break

                        all_states.extend(response.dst)
                        self.trace.append(response)
                        self.num_expansions += 1

                    self.nodes = {n.goal: n for n in all_states if isinstance(n, InternalNode)}

                    if not root.status == Status.PROVED:
                        logger.info("Proof not replicated.")
                        self.total_time = time.monotonic() - time_start
                        return False
                    else:
                        logger.info("Generating additional data from proof nodes")
                        all_states = [n for n in all_states if isinstance(n, InternalNode)]
                        self.proof_path = all_states
                        self._step(env)
                        return True
                except Exception as e:
                    raise Exception(e)

        except Exception as e:
            if root:
                logger.warning(f"Error in search {e}")
                traceback.print_exc()
                root.status = Status.FAILED
                self.log_error(str(e), get_thm_name(self.env_name, env.thm))
            else:
                raise Exception(e)


class DistributedReplay:
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

            prover_pool.append(ray.remote(num_cpus=config.cpu_per_prover)(ReplayProver).remote(
                tac_model=tac_model, timeout=config.env_timeout,
                directory=config.exp_config.directory, env_name=config.env_config.env, iteration=iteration
            ))

        else:
            for i in range(config.logical_gpus):
                tac_model = get_tac_model(config.tac_model, device)

                prover_pool.extend(
                    [ray.remote(num_gpus=config.gpu_per_prover, num_cpus=config.cpu_per_prover)(ReplayProver).remote(
                        tac_model=tac_model, timeout=config.env_timeout,
                        directory=config.exp_config.directory, env_name=config.env_config.env, iteration=iteration
                    ) for _ in range(config.provers_per_gpu)])

        self.prover_pool = ActorPool(prover_pool)

        return

    def search_unordered(self, theorems, proofs, env='leandojo'):
        try:
            env_func = get_env(env)
            results_ = self.prover_pool.map_unordered(
                lambda p, thm: p.search.remote(env_func(thm, self.total_timeout), proofs[get_thm_name(env, thm[1])]),
                theorems,
            )

            proven = 0
            for i, res in enumerate(results_):
                if res:
                    proven += 1

            return proven

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)


def _load_data(data_path: str, normalize_tactics: bool):
    proofs = {}
    for thm in tqdm(json.load(open(data_path))):
        proof = []
        for tac in thm["traced_tactics"]:
            if "annotated_tactic" in tac:
                tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
            else:
                tactic = format_tactic(tac["tactic"], [], normalize_tactics)
            # if not self.keep_marks:
            #     tactic = remove_marks(tactic)

            proof.append(tactic)

        proofs[thm["full_name"]] = proof

    logger.info(f"{len(proofs)} examples loaded")
    return proofs


@hydra.main(config_path="../../configs")
def main(config) -> None:
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    theorems = get_theorems(config.env_config, [])

    set_logger(config.log_level)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Config: {config}")

    if config.shuffle:
        random.shuffle(theorems)

    # todo only for LeanDojo
    proofs = _load_data(config.proof_path, True)

    valid_thms = []
    for thm in tqdm(theorems):
        if get_thm_name('leandojo', thm[1]) not in proofs:
            logger.warning(f'No proof found for {thm[1]}')
        elif proofs[get_thm_name('leandojo', thm[1])]:
            valid_thms.append(thm)

    prover = DistributedReplay(config, 0)

    logger.info(f'Attempting {len(theorems)} proofs..')

    num_proven = prover.search_unordered(valid_thms, env=config.env_config.env, proofs=proofs)

    ray.shutdown()


if __name__ == '__main__':
    main()
