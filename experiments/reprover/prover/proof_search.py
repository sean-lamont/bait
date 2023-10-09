"""Proof search using best-first search.
"""
import os
import pickle
import sys
import ray
import time
import heapq
import torch
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    # LeanError,
    TacticState,
    # LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
# from lean_dojo.interaction.dojo import TacticError
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool
from lean_dojo.constants import LEAN3_DEPS_DIR, LEAN4_DEPS_DIR

from common import zip_strict
from prover.search_tree import *
from generator.model import RetrievalAugmentedGenerator


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]
    tree: Node

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.position = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        try:
            with Dojo(thm, hard_timeout=600 + self.timeout) as (dojo, init_state):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        self._best_first_search()
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            # todo add additional results for training on synthetic data
            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                tree=self.root,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)

            if proof:
                with open(f"traces_orig/proven_{thm.full_name}_{time.strftime('%Y-%m-%d_%H:%M')}.pk", "wb") as f:
                    logger.info(f'Saving proof result of {thm.full_name}')
                    pickle.dump(result, f)
            else:
                with open(f"traces_orig/failed_{thm.full_name}_{time.strftime('%Y-%m-%d_%H:%M')}.pk", "wb") as f:
                    logger.info(f'Saving proof result of {thm.full_name}')
                    pickle.dump(result, f)

            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

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

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = self._generate_tactics(ts)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.

        results = [
            self._run_tactic(search_node, tactic, logprob)
            for tactic, logprob in suggestions
        ]

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            if self.theorem.repo.uses_lean3:
                path = os.path.join(LEAN3_DEPS_DIR, self.theorem.repo.name, path)
            elif self.theorem.repo.is_lean:
                raise NotImplementedError
                path = os.path.join(LEAN4_DEPS_DIR, "lean4", path)
            else:
                path = os.path.join(LEAN4_DEPS_DIR, self.theorem.repo.name, path)

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.position,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float) -> Edge:
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                # TacticError,
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                # TacticError,
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()


@ray.remote
class CpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""

    def __init__(
        self,
        ckpt_path: str,
        indexed_corpus_path: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        tac_gen = RetrievalAugmentedGenerator.load(
            ckpt_path, device=torch.device("cpu"), freeze=True
        )
        if tac_gen.retriever is not None:
            if indexed_corpus_path is not None:
                tac_gen.retriever.load_corpus(indexed_corpus_path)
            tac_gen.retriever.reindex_corpus(batch_size=32)
        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
        )


# set to e.g. 0.5 to share same gpu across CPUs
@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
        self,
        ckpt_path: str,
        indexed_corpus_path: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        tac_gen = RetrievalAugmentedGenerator.load(
            ckpt_path, device=torch.device("cuda"), freeze=True
        )

        if tac_gen.retriever is not None:
            if indexed_corpus_path is not None:
                tac_gen.retriever.load_corpus(indexed_corpus_path)
            tac_gen.retriever.reindex_corpus(batch_size=32)

        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
        )


class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        ckpt_path: str,
        indexed_corpus_path: Optional[str],
        num_cpus: int,
        with_gpus: bool,
        timeout: int,
        num_sampled_tactics: int,
        debug: Optional[bool] = False,
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

            self.prover = BestFirstSearchProver(
                tac_gen, timeout, num_sampled_tactics, debug
            )
            return

        if with_gpus:
            logger.info(f"Launching {num_cpus} GPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=num_cpus)
            provers = [
                GpuProver.remote(
                    ckpt_path,
                    indexed_corpus_path,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]
        else:
            logger.info(f"Launching {num_cpus} CPU workers.")
            ray.init(num_cpus=num_cpus, num_gpus=0)
            provers = [
                CpuProver.remote(
                    ckpt_path,
                    indexed_corpus_path,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> List[SearchResult]:
        # theorems = theorems[120:]
        # positions = positions[120:]
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x[0], x[1]),
                    zip_strict(theorems, positions),
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results
