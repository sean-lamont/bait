import time
from typing import Tuple

from loguru import logger
from tqdm import tqdm

from data.holist.utils import normalization_lib
from data.holist.utils.io_util import load_theorem_database_from_file
from data.holist.utils.theorem_fingerprint import Fingerprint
from environments.holist import proof_assistant, proof_assistant_pb2
from experiments.holist.utils import error
from refactor.proof_node import *


class EnvInitError(Exception):
    pass


# todo abstract environment class with init, enter, exit, run_tactic, retrieve_premises


def setup_prover(theorem_database: proof_assistant_pb2.TheoremDatabase):
    """Starts up HOL and seeds it with given TheoremDatabase."""

    logger.info('Setting up and registering theorems with proof assistant...')
    proof_assistant_obj = proof_assistant.ProofAssistant()

    for thm in tqdm(theorem_database.theorems):
        response = proof_assistant_obj.RegisterTheorem(
            proof_assistant_pb2.RegisterTheoremRequest(theorem=thm))

        if response.HasField('error_msg') and response.error_msg:
            logger.error('Registration failed for %d with: %s' %
                         (response.fingerprint, response.error_msg))

    logger.info('Proof assistant setup done.')

    return proof_assistant_obj


# todo normalization lib? or do that in tac_gen. Seems like only normalised before running tactic
def _thm_string(thm: proof_assistant_pb2.Theorem):
    # Turn theorem into a string

    # if len(thm.hypotheses) > 1:
    #     return thm.hypotheses[0] + '\n'.join([str(hyp) for hyp in thm.hypotheses[1:]]) + '\n|-' + str(thm.conclusion)
    # elif (thm.hypotheses):
    #     return thm.hypotheses[0] + '\n|-' + str(thm.conclusion)
    # else:
    #     return '|-' + str(thm.conclusion)

    # for now, just use normalized conclusion from original HOList model

    return normalization_lib.normalize(thm).conclusion

'''

Environment Wrapper over HOList. Adds premise retrieval and processing of proof tree

'''


class HOListEnv:
    def __init__(self, thm, timeout):
        self.timeout = timeout
        self.environment_time = 0

        # dictionary mapping goals to the corresponding HOList Theorem
        self.node_map = {}

        self.thm, self.database_name = thm

        self.theorem_database = load_theorem_database_from_file(
            str(self.database_name))

        self.premises = self.retrieve_premises()

    def __enter__(self):
        try:
            # todo setup docker container and multiple instances?
            self.hol_wrapper = setup_prover(self.theorem_database)

            logger.info('HOList dependencies initialization complete.')

        except Exception as e:
            raise EnvInitError(e)
        root_goal = _thm_string(self.thm)
        root = InternalNode(goal=root_goal, cumulative_logprob=0.0)

        # dict for goal string to Node, and HOList Theorem
        self.node_map[root_goal] = (root, self.thm)

        return self, root

    def __exit__(self, exc_type, exc_val, exc_tb):
        # todo
        pass

    def retrieve_premises(self):
        fp = Fingerprint(self.thm)

        thm_index_by_fingerprint = {
            Fingerprint(thm): i
            for (i, thm) in enumerate(self.theorem_database.theorems)
        }

        thm_number = thm_index_by_fingerprint.get(fp)

        return self.theorem_database.theorems[:thm_number]

    def run_tactic(self, node: Tuple[InternalNode, float], tactic: Tuple[str, float]):  # -> Tuple[Edge, List]:
        t0 = time.monotonic()

        tactic, tac_logprob = tactic

        node, goal_logprob = node

        node, theorem = self.node_map[node.goal]

        # todo some kind of tactic pre-processing?
        # tactic_ = remove_marks(tactic)

        holist_request = proof_assistant_pb2.ApplyTacticRequest(tactic=tactic, goal=theorem)

        failed = False
        result_node = []
        try:
            response = self.hol_wrapper.ApplyTactic(holist_request)
        except error.StatusNotOk as exception:
            response = EnvironmentError(message=exception.message)
            result_node = ErrorNode(response)
            result = [result_node]
            failed = True

            # From original implementation:
            # Sometimes, rarely, the prover gets into in which it stops
            # communicating and eventually requests hang. However we
            # can bail out before that happen and can prevent the whole
            # program to hang for a long time.

            if str(exception).startswith('Communication') and str(exception).endswith(
                    'failed.'):
                raise exception

        elapsed = time.monotonic() - t0

        self.environment_time += elapsed

        if response.HasField('error') and not failed:
            response = EnvironmentError(message=f'Tactic Application Error')
            result_node = ErrorNode(response)
            result = [result_node]
        else:
            assert response.HasField('goals')
            new_goals = list(response.goals.goals)

            def is_same_expr(t1, t2):
                return t1.conclusion == t2.conclusion and t1.hypotheses == t2.hypotheses

            if len(new_goals) == 1 and is_same_expr(holist_request.goal, new_goals[0]):
                response = TreeError('No change from tactic')
                result_node = ErrorNode(response)
                result = [result_node]

            elif theorem in new_goals:
                response = TreeError('Selected goal remains in response')
                result_node = ErrorNode(response)
                result = [result_node]

            # if no new goals, goal is proven
            elif not new_goals:
                result_node = ProofFinishedNode(GoalFinished())
                result = [result_node]
            # new goals are present
            else:
                result = []
                for i, goal in enumerate(new_goals):

                    thm = proof_assistant_pb2.Theorem(
                        hypotheses=goal.hypotheses,
                        conclusion=goal.conclusion,
                        pretty_printed=goal.pretty_printed,
                        tag=proof_assistant_pb2.Theorem.GOAL)

                    goal = _thm_string(thm)

                    # Treat cycles as error nodes
                    if goal in node.ancestors:
                        response = TreeError('Tactic Creates cycle')
                        result_node = ErrorNode(response)
                        result = [result_node]
                        break
                    if goal in self.node_map:
                        result_node, thm = self.node_map[goal]
                    else:
                        result_node = InternalNode(
                            goal=goal,
                            cumulative_logprob=tac_logprob + node.cumulative_logprob,
                            depth=node.depth + 1
                        )

                        self.node_map[goal] = result_node, thm

                    # todo add below to updown/search processing?
                    # This will add the parent context (any goals required to prove the parent)
                    # as well as other siblings from the current result.
                    sib_context = {_thm_string(goal_) for goal_ in new_goals if
                                   _thm_string(goal_) != goal}
                    if node.context:
                        cur_context = [ctx | sib_context for ctx in node.context]
                    else:
                        cur_context = [sib_context]

                    result_node.add_context(cur_context)

                    # Add ancestors for detecting cycles
                    result_node.add_ancestors(node.ancestors | {node.goal})

                    result.append(result_node)

        # self-loop sanity check (should never happen)
        if result_node == node:
            logger.error(f'Self loop found')
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result, tac_logprob=tac_logprob, goal_logprob=goal_logprob,
                    time=elapsed)

        if node.out_edges:
            node.out_edges = node.out_edges + [edge]
        else:
            node.out_edges = [edge]

        for result_node in result:
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(edge)

        return edge
