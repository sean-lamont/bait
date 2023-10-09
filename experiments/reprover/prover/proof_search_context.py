"""Proof search using best-first search.
"""
import heapq
import os
import pickle
import sys
import time
from typing import Dict

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

from common import zip_strict
from generator.model import RetrievalAugmentedGenerator
from prover.search_tree_context import *


# what to record:
# - All proving edges
# - Tactic state before and after from proving edges
# - visit count?


# for training:
# before, after as seq2seq objective for tactics in a proof (either subgoal proof, or original proof)
# P(PROVABLE | g, CRITIC), add CRITIC token to a state, and restrict output to PROVABLE | UNPROVABLE, then take the models
# probability of the provable token, with CE loss. Ground truth given by 1 if proven, or estimate with visit count and
# W(g, t*) / N (g, t*) with visit count N, goal g and best tactic t*. N(g, t*) is just number of descendants of g.
# W(g, t*) is
@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


# todo new prover needs a goal selection model (similar to tac_gen)
class BestFirstSearchProver:
    """A prover based on all candidate paths from a proof tree."""

    def __init__(
            self,
            tac_gen,  # A given tactic generator.
            # goal_model,
            timeout: int,
            num_sampled_tactics: int,
            debug: bool,
    ) -> None:

        self.tac_gen = tac_gen
        # self.goal_model = goal_model
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
                    goal=init_state.pp,
                    goal_num=0,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state.pp: self.root}
                self.priority_queue = [self.root]

                # dictionary from node to set of children
                # self.paths = {}

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
                nodes=self.nodes
            )

            logger.info(result)

            if proof:
                with open(f"traces/proven_{thm.full_name}_{time.strftime('%Y-%m-%d_%H:%M')}.pk", "wb") as f:
                    logger.info(f'Saving proof result of {thm.full_name}')
                    pickle.dump(result, f)
            else:
                with open(f"traces/failed_{thm.full_name}_{time.strftime('%Y-%m-%d_%H:%M')}.pk", "wb") as f:
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

        # Don't run step if ancestor has already been proven
        for a in search_node.ancestors:
            if self.nodes[a].status == Status.PROVED:
                logger.debug(f'Ancestor proven, skipping this node {search_node}')
                return

        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        ts = search_node.goal

        # todo could parallelise here by popping multiple search_nodes up to a limit
        suggestions = self._generate_tactics(ts)

        # todo have results and _generate_tactics asynchronous: if remaining suggestions,
        # todo then run them, and while they're running keep generating suggestions
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

        if node.goal_num != 0:
            # ensure the tactic is applied to the correct goal in the surrogate state
            tactic_ = f'tactic.rotate_left {node.goal_num}, ' + tactic
        else:
            tactic_ = tactic

        response = self.dojo.run_tac(node.state, tactic_)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        node.visit_count += 1

        result_node = []

        if type(response) in (
                TacticError,
                # LeanError,
                TimeoutError,
                ProofGivenUp,
        ):
            result_node = ErrorNode(response)
            result = [result_node]

        elif isinstance(response, ProofFinished):
            result_node = ProofFinishedNode(GoalFinished())
            result = [result_node]
        else:
            assert isinstance(response, TacticState)

            response_goals = [g for g in response.pp.split("\n\n")]
            prev_goals = [g for g in node.state.pp.split("\n\n")]
            new_goals = [g for g in response_goals if g not in prev_goals]

            # Ensure that the selected goal was actually worked on
            # i.e. no additional rotates etc. in sampled tactic, no self cycles
            if node.goal in response_goals:
                response = TreeError('Selected goal remains in response')
                result_node = ErrorNode(response)
                result = [result_node]

            # no new goals, and previous goal not present, so selected goal is proven
            elif not new_goals:
                # sanity checks
                if response.num_goals >= node.state.num_goals:
                    logger.info(
                        f'edge case: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    response = TreeError(
                        f'edge case 1: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    result_node = ErrorNode(response)
                elif not all([g in prev_goals for g in response_goals]):
                    logger.info(
                        f'edge case 2: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    response = TreeError(
                        f'edge case 2: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    result_node = ErrorNode(response)
                # if more than one goal proven by the tactic application
                elif response.num_goals != node.state.num_goals - 1:
                    # logger.info(
                    #     f'edge case 3: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    response = TreeError(
                        f'edge case 3: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    result_node = ErrorNode(response)
                else:
                    result_node = ProofFinishedNode(GoalFinished())

                result = [result_node]
            # new goals are present, replacing old goal
            else:
                result = []
                for i, goal in enumerate(new_goals):
                    # Treat cycles as error nodes
                    if goal in node.ancestors:
                        response = TreeError('Tactic Creates cycle')
                        result_node = ErrorNode(response)
                        result = [result_node]
                        break
                    if goal in self.nodes:
                        result_node = self.nodes[goal]
                    else:
                        result_node = InternalNode(
                            state=response,
                            goal=goal,
                            goal_num=i,
                            cumulative_logprob=logprob + node.cumulative_logprob,
                        )

                    # Add context to results.
                    # This will add the parent context (any goals required to prove the parent)
                    # as well as other siblings from the current result.
                    sib_context = {goal_ for goal_ in new_goals if goal_ != goal}
                    if node.context:
                        cur_context = [ctx | sib_context for ctx in node.context]
                    else:
                        cur_context = [sib_context]
                    result_node.add_context(cur_context)

                    # add ancestors for detecting cycles
                    result_node.add_ancestors(node.ancestors | {node.goal})
                    result.append(result_node)

        # self-loop sanity check
        if result_node == node:
            logger.info(f'Self loop found')
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result, logprob=logprob, time=elapsed)

        for result_node in result:
            # Record the new node and add it to the search queue.
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(edge)
                self.nodes[result_node.goal] = result_node
                node.children = node.children | {result_node.goal}
            # Don't search proved/explored/queued nodes
            if result_node.status == Status.OPEN and not result_node.is_explored and result_node not in self.priority_queue:
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        return edge

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        # print ("here")
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            if node.is_explored:
                pass
        #
        # for goal, node in self.nodes.items():
        #     # if isinstance(node, ProofFinished):
        #     #     assert isinstance(node, ProofFinishedNode)
        #     #     assert node not in self.priority_queue
        #     #     assert self.root.status == Status.PROVED
        #     # elif type(goal) in (
        #     #         LeanError,
        #     #         # TacticError,
        #     #         TimeoutError,
        #     #         ProofGivenUp,
        #     # ):
        #         assert isinstance(node, ErrorNode)
        #         assert node not in self.priority_queue
        #     else:
        #         assert isinstance(node, InternalNode)
        #
        #         if node.is_explored:
        #             assert node not in self.priority_queue
        #         else:
        #             assert node in self.priority_queue
        #
        #         node.check_invariants()


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
            # goal_model = None
            self.prover = BestFirstSearchProver(
                tac_gen, timeout, num_sampled_tactics, debug
                # tac_gen, goal_model, timeout, num_sampled_tactics, debug
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
        # theorems = theorems[120+96:]
        # positions = positions[120+96:]
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
