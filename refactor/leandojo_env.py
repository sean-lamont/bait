import time
from loguru import logger

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
    TimeoutError,
    TacticState,
    ProofGivenUp
)

# todo wrap in a context manager over LeanDojo, as below
# https://stackoverflow.com/questions/31189526/what-is-the-pythonic-way-to-inherit-context-manager

class LeanDojoEnv:
    def __init__(self):
        # need a dictionary mapping goals to their state
        self.goal_map = {}
        # self.dojo = ..
        self.environment_time = 0
        self.nodes = {}

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float):  # -> Tuple[Edge, List]:
        t0 = time.monotonic()

        goal_num, state = self.goal_map[node.goal]

        if goal_num != 0:
            # ensure the tactic is applied to the correct goal in the surrogate state
            tactic_ = f'tactic.rotate_left {goal_num}, ' + tactic
        else:
            tactic_ = tactic

        response = self.dojo.run_tac(state, tactic_)

        elapsed = time.monotonic() - t0

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
                            goal=goal,
                            cumulative_logprob=logprob + node.cumulative_logprob,
                            depth=node.depth + 1
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

                # if new_nodes:
                #     # compute provable_score/up_score for new internal nodes
                #     node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]
                #
                #     t1 = time.monotonic()
                #     scores = self.goal_model.batch_generate(node_goals)
                #     self.goal_time += time.monotonic() - t1
                #
                #     for i, node_ in enumerate(new_nodes):
                #         node_.provable_score = scores[i] + (node_.depth * math.log(0.95))
                #         node_.up_score = node_.provable_score

        # self-loop sanity check
        if result_node == node:
            logger.debug(f'Self loop found')
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result, logprob=logprob, time=elapsed)

        # for result_node in result:
        #     # Record the new node and add it to the search queue.
        #     if isinstance(result_node, InternalNode):
        #         result_node.in_edges.append(edge)
        #         self.nodes[result_node.goal] = result_node
        #         node.children = node.children | {result_node.goal}

            # Don't search proved/explored/queued nodes
            # if result_node.status == Status.OPEN and not result_node.is_explored and result_node not in self.priority_queue:
            #     heapq.heappush(self.priority_queue, result_node)  # type: ignore

        return edge
