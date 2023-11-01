import os
import time
from typing import Tuple

from loguru import logger

from lean_dojo.constants import LEAN3_DEPS_DIR, LEAN4_DEPS_DIR
from refactor.proof_node import *

from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    ProofFinished,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
    TacticError,
    # LeanError,
    TimeoutError,
    TacticState,
    ProofGivenUp
)

'''

Environment Wrapper over LeanDojo. Adds premise retrieval and processing of proof tree

'''
class LeanDojoEnv:
    def __init__(self, repo, thm, pos, timeout):
        # need a dictionary mapping goals to their state
        self.timeout = timeout
        self.environment_time = 0
        self.node_map = {}
        self.thm = thm
        self.repo = repo
        self.pos = pos

    def __enter__(self):
        self.dojo, init_state = Dojo(self.thm, hard_timeout=600 + self.timeout).__enter__()

        root = InternalNode(goal=init_state.pp, cumulative_logprob=0.0)

        self.node_map[init_state.pp] = (0, init_state, root)

        return self, root

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dojo.__exit__(exc_type, exc_val, exc_tb)

    def retrieve_premises(self):
        path = str(self.thm.file_path)

        if self.thm.repo != self.repo:
            if self.thm.repo.uses_lean3:
                path = os.path.join(LEAN3_DEPS_DIR, self.thm.repo.name, path)
            elif self.thm.repo.is_lean:
                raise NotImplementedError
            else:
                path = os.path.join(LEAN4_DEPS_DIR, self.thm.repo.name, path)

        return path, self.thm, self.pos

    def run_tactic(self, node: InternalNode, tactic: Tuple[str, float]):  # -> Tuple[Edge, List]:
        t0 = time.monotonic()

        tactic, logprob = tactic

        try:
            goal_num, state, _ = self.node_map[node.goal]
        except:
            logger.warning(f'wtf: {self.node_map, node.goal}')

        if goal_num != 0:
            # ensure the tactic is applied to the correct goal in the surrogate state
            tactic_ = f'tactic.rotate_left {goal_num}, ' + tactic
        else:
            tactic_ = tactic

        response = self.dojo.run_tac(state, tactic_)

        elapsed = time.monotonic() - t0

        self.environment_time += elapsed

        # node.visit_count += 1

        result_node = []

        if type(response) in (
                TacticError,
                # LeanError,
                TimeoutError,
                ProofGivenUp,
        ):
            response = EnvironmentError(message=response.error)
            result_node = ErrorNode(response)
            result = [result_node]

        elif isinstance(response, ProofFinished):
            result_node = ProofFinishedNode(GoalFinished())
            result = [result_node]
        else:
            assert isinstance(response, TacticState)
            response_goals = [g for g in response.pp.split("\n\n")]
            prev_goals = [g for g in state.pp.split("\n\n")]
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
                if response.num_goals >= state.num_goals:
                    logger.info(
                        # e.g response contains two identical goals, which was in prev_goals
                        f'edge case: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {state}')
                    response = TreeError(
                        f'edge case 1: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {state}')
                    result_node = ErrorNode(response)
                elif not all([g in prev_goals for g in response_goals]):
                    logger.info(
                        f'edge case 2: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {state}')
                    response = TreeError(
                        f'edge case 2: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {state}')
                    result_node = ErrorNode(response)
                # if more than one goal proven by the tactic application
                elif response.num_goals != state.num_goals - 1:
                    # logger.info(
                    #     f'edge case 3: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {node.state}')
                    response = TreeError(
                        f'edge case 3: {tactic_}, prev_goals: {prev_goals}, response: {response}, state: {state}')
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
                    if goal in self.node_map:
                        goal_num, _, result_node = self.node_map[goal]
                    else:
                        result_node = InternalNode(
                            goal=goal,
                            cumulative_logprob=logprob + node.cumulative_logprob,
                            depth=node.depth + 1
                        )

                        self.node_map[goal] = (i, response, result_node)

                    # todo add below to search processing?
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

        # self-loop sanity check
        if result_node == node:
            logger.debug(f'Self loop found')
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result, logprob=logprob, time=elapsed)

        return edge