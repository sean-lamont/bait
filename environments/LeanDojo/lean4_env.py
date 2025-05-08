import os
import time
from typing import Tuple

from lean_dojo import (
    Dojo,
    ProofFinished,
    DojoTacticTimeoutError as TimeoutError,
    TacticState,
    ProofGivenUp,
    LeanError
)

from lean_dojo.constants import LEAN4_PACKAGES_DIR

from experiments.end_to_end.common import remove_marks
from experiments.end_to_end.proof_node import *


class EnvInitError(Exception):
    pass


'''

Environment Wrapper over LeanDojo Lean 4.

'''


class Lean4Env:
    def __init__(self, thm, timeout):
        self.timeout = timeout
        self.environment_time = 0
        # dictionary mapping goals to their state
        self.node_map = {}
        self.repo, self.thm, self.pos = thm

        self.premises = self.retrieve_premises()

    def __enter__(self):
        try:
            self.dojo, init_state = Dojo(self.thm, timeout=600 + self.timeout).__enter__()
            # self.dojo, init_state = Dojo(self.thm, timeout=600 + self.timeout, additional_imports=["Mathlib.Tactic"]).__enter__()
        except Exception as e:
            raise EnvInitError(e)

        root = InternalNode(goal=init_state.pp, cumulative_logprob=0.0)

        self.node_map[init_state.pp] = (init_state, root)

        return self, root

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dojo.__exit__(exc_type, exc_val, exc_tb)

    def _cleanup(self):
        pass
        # self.dojo._cancel_timer()
        # self.dojo._cleanup()

    def retrieve_premises(self):
        path = str(self.thm.file_path)

        if self.thm.repo != self.repo:
            path = os.path.join(LEAN4_PACKAGES_DIR, self.thm.repo.name, path)
        return path, self.thm, self.pos

    def run_tactic(self, node: Tuple[InternalNode, float], tactic: Tuple[str, float]):  # -> Tuple[Edge, List]:
        t0 = time.monotonic()

        tactic, tac_logprob = tactic

        node, goal_logprob = node

        state, _ = self.node_map[node.goal]

        tactic_ = remove_marks(tactic)

        response = self.dojo.run_tac(state, tactic_)

        elapsed = time.monotonic() - t0

        self.environment_time += elapsed

        if type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
        ):
            response = EnvironmentError(message=response.error if type(response) == LeanError else 'Timeout')
            result_node = ErrorNode(response)
            result = [result_node]

        elif isinstance(response, ProofFinished):
            result_node = ProofFinishedNode(GoalFinished())
            result = [result_node]
        else:
            assert isinstance(response, TacticState)
            goal = response.pp
            # Treat cycles as error nodes
            if goal in node.ancestors or goal == node.goal:
                response = TreeError('Tactic Creates cycle')
                result_node = ErrorNode(response)
                result = [result_node]
            elif goal in self.node_map:
                _, result_node = self.node_map[goal]

                result = [result_node]
            else:
                result_node = InternalNode(
                    goal=goal,
                    cumulative_logprob=tac_logprob + node.cumulative_logprob,
                    depth=node.depth + 1
                )

                self.node_map[goal] = (response, result_node)

                result = [result_node]

        if result_node == node:
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        edge = Edge(tactic=tactic, src=node, dst=result, tac_logprob=tac_logprob, goal_logprob=goal_logprob,
                    time=elapsed)

        node.add_edge(edge)

        return edge
