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
from goal_model_step.model import StepGoalModel
from updown_step.search_tree_updown import *


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
    # goal_time: float
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
            goal_model,
            timeout: int,
            num_sampled_tactics: int,
            debug: bool,
    ) -> None:

        self.tac_gen = tac_gen
        self.goal_model = goal_model
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.goal_time = 0.0

        self.total_time = None

        self.dir = f'traces_{time.strftime("%Y-%m-%d_%H:%M")}'
        os.makedirs(self.dir, exist_ok=True)


    def search(
            self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.position = pos
        self.actor_time = 0.0
        self.goal_time = 0.0
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
                goal_time=self.goal_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
                nodes=self.nodes
            )

            logger.info(result)

            logger.info(f'Saving proof result of {thm.full_name}')
            if proof:
                with open(f"{self.dir}/proven_{thm.full_name}.pk", "wb") as f:
                    pickle.dump(result, f)
            else:
                with open(f"{self.dir}/failed_{thm.full_name}.pk", "wb") as f:
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

    def _get_best_goal(self):
        if len(self.nodes) == 1:
            return self.root

        best_score = -math.inf
        best_node = None

        for goal, node in self.nodes.items():
            if node.is_explored or node.ignore:
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

        return best_node

    def _step(self):
        """

        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.

        """

        # Search the node with highest priority.
        # search_node = heapq.heappop(self.priority_queue)
        search_node = self._get_best_goal()

        if search_node is None:
            logger.info(f'No valid nodes selected, returning failure')
            self.root.status = Status.FAILED
            return

        # Don't run step if ancestor has already been proven
        for a in search_node.ancestors:
            if self.nodes[a].status == Status.PROVED:
                logger.debug(f'Ancestor proven, skipping this node {search_node}')
                search_node.ignore = True
                return

        # logger.info(f"Expanding node: {search_node}")

        # if self.debug:
        #     assert all(
        #         search_node.priority >= node.priority for node in self.priority_queue
        #     )

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

        # results = []
        # all_new_nodes = []
        # node_scores = torch.tensor([]).cuda()
        # for tactic, logprob in suggestions:
        #     edge, new_nodes = self._run_tactic(search_node, tactic, logprob)
        #     results.append(edge)
        #
        #     if new_nodes:
        #         all_new_nodes.extend(new_nodes)
        #         # compute provable_score/up_score for new internal nodes
        #         node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]
        #
        #         t1 = time.monotonic()
        #         node_scores = torch.cat([node_scores, self.goal_model.batch_generate(node_goals)], dim=0)
        #         self.goal_time += time.monotonic() - t1
        #
        # # todo ensure these are all updated before out_edges is
        # for i, node_ in enumerate(all_new_nodes):
        #     node_.provable_score = node_scores[i]
        #     node_.up_score = node_scores[i]
        # #
        #
        # assert all(node_.provable_score != math.inf for node_ in all_new_nodes)

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results

        # run up_step to update the proof state
        search_node.up_score = -math.inf
        up_step(search_node)

        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )

        # self.check_invariants()

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

        # logger.info(f'Gen time {time.monotonic() - t0}')

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float):# -> Tuple[Edge, List]:
        t0 = time.monotonic()

        if node.goal_num != 0:
            # ensure the tactic is applied to the correct goal in the surrogate state
            tactic_ = f'tactic.rotate_left {node.goal_num}, ' + tactic
        else:
            tactic_ = tactic

        response = self.dojo.run_tac(node.state, tactic_)

        # test

        elapsed = time.monotonic() - t0

        # logger.info(f'Env time {elapsed}')

        self.environment_time += elapsed

        new_nodes = []

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
                        # e.g response contains two identical goals, which was in prev_goals
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
                        new_nodes.append(result_node)

                    # This will add the parent context (any goals required to prove the parent)
                    # as well as other siblings from the current result.
                    sib_context = {goal_ for goal_ in new_goals if goal_ != goal}
                    if node.context:
                        cur_context = [ctx | sib_context for ctx in node.context]
                    else:
                        cur_context = [sib_context]
                    result_node.add_context(cur_context)

                    # Add ancestors for detecting cycles
                    result_node.add_ancestors(node.ancestors | {node.goal})
                    result.append(result_node)

                if new_nodes:
                    # compute provable_score/up_score for new internal nodes
                    node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]

                    t1 = time.monotonic()
                    scores = self.goal_model.batch_generate(node_goals)
                    self.goal_time += time.monotonic() - t1

                    for i, node_ in enumerate(new_nodes):
                        node_.provable_score = scores[i]
                        node_.up_score = scores[i]

        # self-loop sanity check
        if result_node == node:
            logger.debug(f'Self loop found')
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
            # if result_node.status == Status.OPEN and not result_node.is_explored and result_node not in self.priority_queue:
            #     heapq.heappush(self.priority_queue, result_node)  # type: ignore

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
            goal_path: str,
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

        goal_model = StepGoalModel.load(goal_path, device=torch.device("cpu"), freeze=True)

        super().__init__(
            tac_gen,
            goal_model,
            timeout,
            num_sampled_tactics,
            debug,
        )


# set to 1 / (gpu_mem //  required_mem)
@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
            self,
            ckpt_path: str,
            goal_path: str,
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

        goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)

        super().__init__(
            tac_gen,
            goal_model,
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
            goal_path: str,
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

            goal_model = StepGoalModel.load(goal_path, device=torch.device("cuda"), freeze=True)

            self.prover = BestFirstSearchProver(
                tac_gen, goal_model, timeout, num_sampled_tactics, debug
                # tac_gen, goal_model, timeout, num_sampled_tactics, debug
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
                    goal_path,
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


# For this setup, we assume that one goal will be expanded at a time.
# Therefore the up score will be the maximum over children (if unproved), ignoring the node itself


# For more general setup, where we don't expand all nodes at once:
# Have goal model S(g) used for ranking leaves only, and Ranking model R(g, t) used to score (g,t) pairs
# Both have similar GT
# As tac_gen is the main constraint, selecting goals isn't the bottleneck, but we are bottlenecked by the number of goals which can have tacs generated
# Won't cost much to do additional ranking only for the selected goals. Then ranking is automatically done, and can be used to get better estimates of the score
# Up step will then be maximum over children after tactics are generated, where each child (g,t) is either expanded into nodes, or taken as the predicted score from R(g,t)
# This way, initial node value isn't considered in the maximum after a node has been expanded, preventing it from 'hijacking' the search, and giving a more accurate
# estimate of the node, only accounting for what has been seen in the search attempt


# Assume that all leaf nodes have had their provable_score computed, and set to be their up_score
# also need node.up_score = -math.inf before starting
def up_step(node: Node):
    # redundant?
    # if node.status == Status.PROVED:
    #     node.up_score = 0
    #     for edge in node.in_edges:
    #         up_step(edge.src)

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
                up_step(parent)
