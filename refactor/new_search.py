"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
import uuid
from typing import Dict, Tuple, Any

import ray
import torch
from lean_dojo import (
    Pos,
    Theorem,
    LeanGitRepo,
)
from lean_dojo import is_available_in_cache
from loguru import logger
from ray.util.actor_pool import ActorPool

from experiments.reprover.common import set_logger
from experiments.reprover.common import zip_strict
from experiments.reprover.generator.model import RetrievalAugmentedGenerator
from experiments.reprover.goal_model_step.model import StepGoalModel
from refactor.leandojo_env import LeanDojoEnv
from refactor.proof_node import *


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""
    theorem: str
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)

    # Some statistics during proof search.
    total_time: float
    tac_time: float
    search_time: float
    env_time: float
    num_expansions: int
    num_nodes: int

    # Search trace to reconstruct state+selected goals+probs
    search_trace: Any = field(repr=False)

    # Tac gen trace
    tac_trace: Any = field(repr=False)


# todo add tac/search model train functions, which can be lightning data modules
class EndToEndProver:
    def __init__(self, timeout, search_model, tac_model):
        self.timeout = timeout
        self.search_model = search_model
        self.tac_model = tac_model

        self.total_time = 0
        self.search_time = 0
        self.tac_time = 0
        self.env_time = 0
        self.num_expansions = 0

        self.tac_trace = []

        self.dir = f'traces_{time.strftime("%Y-%m-%d_%H:%M")}'
        os.makedirs(self.dir, exist_ok=True)

    def _process_trace(self, trace):
        with open(f"{self.dir}/{trace.theorem}.pk", "wb") as f:
            pickle.dump(trace, f)
        return

    def _step(self, env):
        t0 = time.monotonic()
        goals = self.search_model.get_goals()
        self.search_time += time.monotonic() - t0

        t0 = time.monotonic()
        tactics = self.tac_model.get_tactics(goals, env)
        if not tactics:
            return
        self.tac_time += time.monotonic() - t0

        t0 = time.monotonic()
        response = env.run_tactic(goals, tactics)
        self.env_time += time.monotonic() - t0

        # record tactic and response in trace
        self.tac_trace.append((tactics, response))

        self.num_expansions += 1

        t0 = time.monotonic()
        self.search_model.process_response(response)
        self.search_time += time.monotonic() - t0

    def search(self, env):
        with torch.no_grad():
            try:
                self._search(env)
            except Exception as e:
                logger.warning(f"Error in search {e}")
                pass

        if self.search_model.root.status == Status.PROVED:
            proof = [e.tactic for e in self.search_model.root.extract_proof()]
        else:
            proof = None

        result = SearchResult(
            theorem=f'{env.thm.full_name}',
            status=self.search_model.root.status,
            proof=proof,
            tree=self.search_model.root,
            nodes=self.search_model.nodes,
            total_time=self.total_time,
            tac_time=self.tac_time,
            search_time=self.search_time,
            env_time=self.env_time,
            num_expansions=self.num_expansions,
            tac_trace=self.tac_trace,
            search_trace=self.search_model.search_trace,
            num_nodes=len(self.search_model.nodes)
        )

        logger.info(f'Result: {result}')

        self._process_trace(result)

        return result

    def _search(self, env) -> None:
        try:
            with env as (env, root):
                time_start = time.monotonic()
                self.search_model.reset(root)
                logger.info(f'Attempting to prove {root}')

                self.search_time = 0
                self.tac_time = 0
                self.env_time = 0
                self.num_expansions = 0
                self.tac_trace = []

                while True:
                    try:
                        self._step(env)
                    except Exception as e:
                        assert time.monotonic() - time_start >= self.timeout, f"Exception not timeout: {e}"

                    self.total_time = time.monotonic() - time_start

                    if self.total_time > self.timeout:
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

    def get_goals(self):
        # Initially will only have the root
        if len(self.nodes) == 1:
            node_goals = ['<extra_id_0>' + self.root.goal]
            scores = self.goal_model.batch_generate(node_goals)

            self.root.provable_score = scores[0]
            self.root.up_score = scores[0]

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

        self.search_trace.append((best_node, best_score.item()))

        return best_node

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

            scores = self.goal_model.batch_generate(node_goals)

            for i, node_ in enumerate(new_nodes):
                node_.provable_score = scores[i] + (node_.depth * math.log(0.95))
                node_.up_score = node_.provable_score

        for result_node in result:
            # Record the new node and add it to the search queue.
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(response)
                self.nodes[result_node.goal] = result_node
                # search_node.children = search_node.children | {result_node.goal}

        if search_node.out_edges:
            search_node.out_edges = search_node.out_edges + [response]
        else:
            search_node.out_edges = [response]

        self._up_step(search_node)
        return


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, env):
        return


class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.remaining_tacs = {}
        self.num_sampled_tactics = num_sampled_tactics

    def _generate_tactics(self, ts: str, env) -> List[Tuple[str, float]]:
        path, theorem, position = env.retrieve_premises()

        tactics = self.tac_model.generate(
            state=ts,
            file_path=path,
            theorem_full_name=theorem.full_name,
            theorem_pos=position,
            num_samples=self.num_sampled_tactics,
        )

        logger.debug(f"Tactic suggestions: {tactics}")
        return tactics

    def get_tactics(self, goals, env):
        search_node = goals
        ts = search_node.goal

        # Get full set of suggestions for goal if it hasn't been computed already
        if ts not in self.remaining_tacs:
            logger.debug(f'Generating tacs for {ts}')
            tacs = self._generate_tactics(ts, env)
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


# set to 1 / (gpu_mem //  required_mem)
@ray.remote(num_gpus=1)
class GpuProver(EndToEndProver):
    """Ray actor for running an instance of `EndToEndProver` on a GPU."""

    def __init__(
            self,
            ckpt_path: str,
            goal_path: str,
            indexed_corpus_path: Optional[str],
            timeout: int,
    ) -> None:

        tac_gen = RetrievalAugmentedGenerator.load(
            ckpt_path, device=torch.device("cuda"), freeze=True
        )

        if tac_gen.retriever is not None:
            if indexed_corpus_path is not None:
                tac_gen.retriever.load_corpus(indexed_corpus_path)
            tac_gen.retriever.reindex_corpus(batch_size=32)

        tac_model = ReProverTacGen(tac_model=tac_gen)

        goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)

        search_model = UpDown(goal_model)

        super().__init__(
            timeout=timeout,
            search_model=search_model,
            tac_model=tac_model
        )


@ray.remote
class CpuProver(EndToEndProver):
    """Ray actor for running an instance of `EndToEndProver` on a GPU."""

    def __init__(
            self,
            ckpt_path: str,
            goal_path: str,
            indexed_corpus_path: Optional[str],
            timeout: int,
    ) -> None:

        tac_gen = RetrievalAugmentedGenerator.load(
            ckpt_path, device=torch.device("cpu"), freeze=True
        )

        if tac_gen.retriever is not None:
            if indexed_corpus_path is not None:
                tac_gen.retriever.load_corpus(indexed_corpus_path)
            tac_gen.retriever.reindex_corpus(batch_size=32)

        goal_model = StepGoalModel.load(goal_path, device=torch.device("cpu"), freeze=True)

        search_model = UpDown(goal_model)

        tac_model = ReProverTacGen(tac_model=tac_gen)

        super().__init__(
            timeout=timeout,
            search_model=search_model,
            tac_model=tac_model
        )


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
    ) -> None:

        self.distributed = num_cpus > 1

        if not self.distributed:
            device = torch.device("cuda") if with_gpus else torch.device("cpu")
            tac_gen = RetrievalAugmentedGenerator.load(
                ckpt_path, device=device, freeze=True
            )
            if tac_gen.retriever is not None:
                assert indexed_corpus_path is not None
                tac_gen.retriever.load_corpus(indexed_corpus_path)

            goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)

            tac_model = ReProverTacGen(tac_model=tac_gen)

            search_model = UpDown(goal_model)

            self.prover = EndToEndProver(
                tac_model=tac_model, search_model=search_model, timeout=timeout
            )
            return

        if with_gpus:
            logger.info(f"Launching {num_cpus} GPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=num_cpus)
            provers = [
                GpuProver.remote(
                    ckpt_path,
                    goal_path,
                    indexed_corpus_path,
                    timeout=timeout,
                )
                for _ in range(num_cpus)
            ]
        else:
            logger.info(f"Launching {num_cpus} CPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=0)
            provers = [
                CpuProver.remote(
                    ckpt_path,
                    goal_path,
                    indexed_corpus_path,
                    timeout=timeout
                )
                for _ in range(num_cpus)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(
            self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> Any:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                # self.prover.search(repo, thm, pos)
                self.prover.search(LeanDojoEnv(repo, thm, pos, 600))
                for thm, pos in zip_strict(theorems, positions)
            ]
        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(LeanDojoEnv(repo, x[0], x[1], 600)),
                    zip_strict(theorems, positions),
                )
            )

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results


def _get_theorems(args) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        args.data_path,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
        data_path: str,
        split: str,
        file_path: Optional[str],
        full_name: Optional[str],
        name_filter: Optional[str],
        num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    cur_url = None
    cur_commit = None
    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
                t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        logger.debug(f'repo {t["url"], t["commit"]}')

        if t['url'] != cur_url or t['commit'] != cur_commit:
            cur_url = t['url']
            cur_commit = t['commit']
            repo = LeanGitRepo(t["url"], t["commit"])

        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))
    theorems = sorted(
        theorems,
        key=lambda t: hashlib.md5(
            (str(t.file_path) + ":" + t.full_name).encode()
        ).hexdigest(),
    )
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint of the tactic generator.",
    )

    parser.add_argument(
        "--goal_path",
        type=str,
        required=True,
        help="Checkpoint of the goal model",
    )

    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    # Randomly generate an experiment ID if not provided.
    if args.exp_id is None:
        args.exp_id = str(uuid.uuid4())

    set_logger(args.verbose)
    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    repo, theorems, positions = _get_theorems(args)

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        args.ckpt_path,
        args.goal_path,
        args.indexed_corpus_path,
        num_cpus=args.num_cpus,
        with_gpus=args.with_gpus,
        timeout=args.timeout,
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
    if args.exp_id is not None:
        pickle_path = f"{args.exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")


if __name__ == "__main__":
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
