import glob
from loguru import logger
import math
import pickle

import torch
from pymongo import MongoClient
from tqdm import tqdm

from refactor.proof_node import InternalNode, ErrorNode, ProofFinishedNode


def get_traces(path):
    files = glob.glob(path)

    traces = []
    for file in tqdm(files):
        with open(file, "rb") as f:
            trace = pickle.load(f)
            traces.append(trace)
    return traces


def add_goal_data(node, visits):
    steps = node.distance_to_proof

    datum = {
        'goal': node.goal,
        'distance_to_proof': steps,
        'visits': visits[node.goal],
        'local_visits': len(node.out_edges) if node.out_edges else 0,
    }

    return datum


def full_trace_data(trace, visits):
    data = []
    goal_data = []
    visited_goals = set()
    for i, edge in enumerate(trace.trace):
        datum = {
            'iteration': 0,
            'step': i,
            'top_goal': trace.theorem.full_name,
            'goal': edge.src.goal,
            'tactic': edge.tactic,
            'goal_prob': edge.goal_logprob.item() if isinstance(edge.goal_logprob, torch.Tensor) else edge.goal_logprob,
            'tac_prob': edge.tac_logprob,
            'distance_to_proof': edge.distance_to_proof(),
            'visits': edge.visit_count(),
            'time': edge.time, }
        # add children of edge
        if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
            # todo could record error message for e.g. self-correcting proof approach>
            datum['outcome'] = ['Error']
        elif len(edge.dst) == 1 and isinstance(edge.dst[0], ProofFinishedNode):
            datum['outcome'] = ['Proven']
        else:
            outcome = [d.goal for d in edge.dst]
            datum['outcome'] = outcome

            for dst in edge.dst:
                if dst.goal in visits and dst.goal not in visited_goals:
                    goal_data.append({'initial_goal': edge.src.goal,
                                      'subgoal': dst.goal,
                                      'distance_to_proof': edge.distance_to_proof(),
                                      'visits': visits[dst.goal],
                                      'local_visits': len(dst.out_edges) if dst.out_edges else 0
                                      })

                    # only add goals once (ignores multiple parents for initial_goal)
                    visited_goals = visited_goals | {dst.goal}

        data.append(datum)
    return data, goal_data


def add_rand_idx(collection):
    collection.update_many({'rand_idx': {'$exists': False}},
                           [{'$set':
                               {'rand_idx': {
                                   '$function': {
                                       'body': 'function() {return Math.random();}',
                                       'args': [],
                                       'lang': "js"
                                   }
                               }}
                           }]
                           )

    collection.create_index('rand_idx')
    return


# proof length objective, similar to Polu et al.
def transform_goal(goal_datum, max_len=10, visit_threshold=2048):
    proof_len = goal_datum['distance_to_proof']
    if proof_len < max_len:
        return {'goal': goal_datum['goal'], 'target': (max_len + 1) - goal_datum['distance_to_proof']}
    elif proof_len < math.inf:
        return {'goal': goal_datum['goal'], 'target': 1}
    elif goal_datum['visits'] >= visit_threshold:
        return {'goal': goal_datum['goal'], 'target': 0}
    else:
        return None


# binary proven/unproven classification task, approach from HTPS
def transform_goal_proven(goal_datum, visit_threshold=256):
    proof_len = goal_datum['distance_to_proof']
    if proof_len < math.inf:
        return {'goal': goal_datum['goal'], 'target': 1}
    elif goal_datum['visits'] >= visit_threshold:
        return {'goal': goal_datum['goal'], 'target': 0}
    else:
        return None

def transform_subgoal_proven(subgoal_data, visit_threshold=1024):
    proof_len = subgoal_data['distance_to_proof']
    if proof_len < math.inf:
        return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 1}
    elif subgoal_data['visits'] >= visit_threshold:
        return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 0}
    else:
        return None


# create labelled goal provability data with smooth labels for unproven goals, getting closer to 0 as more visits
def transform_goal_smooth(subgoal_data, max_count=4096, min_count=256):
    proof_len = subgoal_data['distance_to_proof']
    if proof_len < math.inf:
        return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 1}
    elif subgoal_data['visits'] >= min_count:
        return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'],
                'target': 0.5 * max(0, (1 - (subgoal_data['visits'] / max_count)))}
    else:
        return None


# new ranking:
# - restricted to max_pairs per goal
# - prioritise rankings: (proven, error), (no error, error)
# - try find pairs with closer length
def rank_edges_new(goal, edges, max_pairs=16):
    valid_edges = [edge for edge in edges if not isinstance(edge.dst[0], ErrorNode)]
    invalid_edges = [('error', edge) for edge in edges if isinstance(edge.dst[0], ErrorNode)]

    # proven edges
    proven_edges = [('proven', edge) for edge in valid_edges if edge.distance_to_proof() < math.inf]

    # non-error edges
    success_non_proven_edges = [('no_error', edge) for edge in valid_edges if edge.distance_to_proof() == math.inf]

    all_edges = sorted(invalid_edges + proven_edges + success_non_proven_edges, key=lambda x: len(x[1].tactic))

    num_proven = len(proven_edges)
    num_failed = len(invalid_edges)
    num_success = len(success_non_proven_edges)

    pairs = []

    while len(pairs) < max_pairs and num_failed >= 0:
        winner = None
        loser = None
        win_type = None
        last_error = None
        for (i, (edge_type, edge)) in enumerate(all_edges):
            if not winner:
                if edge_type == 'proven':
                    winner = edge
                    all_edges.pop(i)
                    num_proven -= 1
                    win_type = edge_type
                elif num_proven <= 0 and edge_type == 'no_error':
                    winner = edge
                    all_edges.pop(i)
                    num_success -= 1
                    win_type = edge_type
                elif edge_type == 'error':
                    last_error = edge

            # will be called once winner is found
            elif edge_type == 'error':
                # nearest error edge will be either last_error or this error, take the closest in tactic length
                if not last_error:
                    loser = edge
                    num_failed -= 1
                    all_edges.pop(i)
                    break
                elif (len(last_error.tactic) - len(winner.tactic)) ** 2 <= (len(edge.tactic) - len(edge.tactic)) ** 2:
                    loser = last_error
                    num_failed -= 1
                    all_edges.pop(i)
                    break
                else:
                    loser = edge
                    num_failed -= 1
                    all_edges.pop(i)
                    break

        if winner and loser and win_type:
            pairs.append((winner, loser, win_type))
        else:
            return

        w_l = [
            {'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic,
             'loser_prob': l.tac_logprob,
             'type': tac_type} for (w, l, tac_type) in pairs]

        rank_collection.insert_many(w_l)

        return


def rank_edges(goal, edges):
    valid_edges = [edge for edge in edges if not isinstance(edge.dst[0], ErrorNode)]
    invalid_edges = [edge for edge in edges if isinstance(edge.dst[0], ErrorNode)]

    # rank all valid_edges above all invalid_edges
    w_l = [{'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic, 'loser_prob': l.tac_logprob,
            'type': 'valid_rank'} for w in valid_edges for l in invalid_edges]

    # from valid_edges, rank proven goals above non_proven valid goals
    proven_edges = [edge for edge in valid_edges if edge.distance_to_proof() < math.inf]
    success_non_proven_edges = [edge for edge in valid_edges if edge.distance_to_proof() == math.inf]

    w_l.extend([{'goal': goal, 'winner': w.tactic, 'winner_prob': w.tac_logprob, 'loser': l.tactic, 'loser_prob': l.tac_logprob,
                 'type': 'proven_rank'} for w in proven_edges for l in success_non_proven_edges])

    # from proven edges, rank based on distance_to_proof, then execution time
    ranked_proofs = sorted(proven_edges, key=lambda x: (x.distance_to_proof(), x.time))

    w_l.extend(
        [{'goal': goal, 'winner': ranked_proofs[i].tactic,
          'winner_prob': ranked_proofs[i].tac_logprob, 'loser': ranked_proofs[j].tactic,
          'loser_prob': ranked_proofs[j].tac_logprob,
          'type': 'time_len_rank'} for i in range(len(ranked_proofs)) for j in
         range(i + 1, len(ranked_proofs))])

    # among successful without proof, rank those that lead to the same outcome based on time only
    for i, edge in enumerate(success_non_proven_edges):
        same_outcome_ranks = []
        for j in range((i + 1), len(success_non_proven_edges)):
            edge_2 = success_non_proven_edges[j]
            edge_1_outcome = [g.goal for g in edge.dst] if isinstance(edge.dst[0], InternalNode) else [
                'Error'] if isinstance(edge.dst[0], ErrorNode) else ['Proven']
            edge_2_outcome = [g.goal for g in edge_2.dst] if isinstance(edge_2.dst[0], InternalNode) else [
                'Error'] if isinstance(edge_2.dst[0], ErrorNode) else ['Proven']
            if set(edge_1_outcome) == set(edge_2_outcome):
                if edge.time < edge_2.time:
                    same_outcome_ranks.append(
                        {'goal': goal, 'winner': edge.tactic, 'winner_prob': edge.tac_logprob, 'loser': edge_2.tactic,
                         'loser_prob': edge_2.tac_logprob, 'type': 'same_outcome'})
                else:
                    same_outcome_ranks.append(
                        {'goal': goal, 'winner': edge_2.tactic, 'winner_prob': edge_2.tac_logprob, 'loser': edge.tactic,
                         'loser_prob': edge.tac_logprob, 'type': 'same_outcome'})

        w_l.extend(same_outcome_ranks)

    if w_l:
        rank_collection.insert_many(w_l)
    return


# todo parameterise, make functions/classes for this, integrate with end to end
if __name__ == '__main__':

    client = MongoClient()
    db = client['lean_e2e']

    # todo how to merge different attempts of same proof?
    # For goal data, if proof length is lower, take that data point. If failed, and visit count higher, replace with that as well
    # I.e. every new attempt, add all new goals, and also update existing goals with above criteria

    # For edge data...
    # Assume all valid/invalid edges are still valid/invalid, then those rankings are fine
    # Rankings from proven/success could be changed if success turns out to be a proof..
    # Rankings within proof could also change, if shorter proof from children is found

    # Seems small/unlikely for this to make much of a difference. Worst case is a longer proof is ranked better than a shorter/slower one

    traces = get_traces('../experiments/runs/leandojo/sample_bestfs_2_2023_11_30/15_32_02/traces/*')

    rank_collection = db['dpo_data']
    goal_collection = db['goal_data']

    goal_len_collection = db['goal_len_task']
    goal_proven_task = db['goal_proven_task']

    goal_labels = db['goal_labels']

    trace_collection = db['edge_data']

    logger.info(f'Adding proof data from traces..')
    for trace in tqdm(traces):
        if isinstance(trace.tree, ErrorNode):
            continue

        nodes = trace.nodes
        nodes[trace.tree.goal] = trace.tree

        updated_visit_count = {node: nodes[node].visit_count for node in nodes.keys()}

        for goal, node in nodes.items():
            for a in node.ancestors:
                updated_visit_count[a] += node.visit_count

        # add full trace data
        trace_data, goal_data = full_trace_data(trace, updated_visit_count)

        if trace_data:
            trace_collection.insert_many(trace_data)
        if goal_data:
            goal_collection.insert_many(goal_data)

        # add edge ranking data for DPO
        for node in nodes.values():
            if node.out_edges:
                rank_edges_new(goal=node.goal, edges=node.out_edges)

    logger.info(f'Adding processed goal data..')

    # add processed data for goal models
    for datum in tqdm(goal_collection.find()):
        proven_data = transform_goal_smooth(datum)
        if proven_data:
            goal_labels.insert_one(proven_data)

    logger.info(f'Adding random field to enable shuffled, streamed dataloading..')
    # add random index for collections used in training (ensures shuffled and ordered data for distributed training)
    add_rand_idx(goal_labels)
    add_rand_idx(rank_collection)
