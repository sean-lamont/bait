import glob
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
        'score': node.up_score.item() if isinstance(node.up_score, torch.Tensor) else node.up_score
    }

    return datum


def full_trace_data(trace):
    data = []
    for i, edge in enumerate(trace.trace):
        datum = {
            'iteration': 0,
            'step': i,
            'top_goal': trace.theorem,
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
        data.append(datum)
    return data


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


# create pairs of winners/losers based on edges from a given goal, and maintain tac probs for each, used for DPO
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


if __name__ == '__main__':

    client = MongoClient()
    db = client['lean_dojo']

    # todo how to merge different attempts of same proof?
    # For goal data, if proof length is lower, take that data point. If failed, and visit count higher, replace with that as well
    # I.e. every new attempt, add all new goals, and also update existing goals with above criteria

    # For edge data...
    # Assume all valid/invalid edges are still valid/invalid, then those rankings are fine
    # Rankings from proven/success could be changed if success turns out to be a proof..
    # Rankings within proof could also change, if shorter proof from children is found
    # Seems small/unlikely for this to make much of a difference. Worst case is a longer proof is ranked better than a shorter/slower one

    traces = get_traces('../experiments/runs/eval_loop/updown_proof_len_2023_11_16/11_47_01/traces/*')

    rank_collection = db['tac_ranks']
    goal_collection = db['goal_data']
    goal_len_collection = db['goal_len_task']
    goal_proven_task = db['goal_proven_task']
    trace_collection = db['edge_data']

    for trace in tqdm(traces):
        if isinstance(trace.tree, ErrorNode):
            continue

        nodes = trace.nodes
        nodes[trace.tree.goal] = trace.tree

        updated_visit_count = {node: nodes[node].visit_count for node in nodes.keys()}

        for goal, node in nodes.items():
            for a in node.ancestors:
                updated_visit_count[a] += node.visit_count

        # add raw data for goal models
        for node in nodes:
            step_datum = add_goal_data(nodes[node], updated_visit_count)
            if step_datum:
                goal_collection.insert_one(step_datum)

        # add full trace data
        trace_collection.insert_many(full_trace_data(trace))

        # add edge ranking data for DPO
        for node in nodes.values():
            if node.out_edges:
                rank_edges(goal=node.goal, edges=node.out_edges)
    #
    # add processed data for goal models
    goal_data = []
    for datum in tqdm(goal_collection.find()):

        goal_data.append({'goal': datum['goal'],
                          'full_visit_count': datum['visits'],
                          'proved': datum['distance_to_proof'] < math.inf})

    # with open('../train_goal.pk', 'wb') as f:
    #     pickle.dump(goal_data[:int(0.9 * len(goal_data))], f)
        
    # with open('../val_goal.pk', 'wb') as f:
    #     pickle.dump(goal_data[int(0.9 * len(goal_data)):], f)

        len_data = transform_goal(datum)
        if len_data:
            goal_len_collection.insert_one(len_data)

        proven_data = transform_goal_proven(datum)
        if proven_data:
            goal_proven_task.insert_one(proven_data)

    # add random index for collections used in training (ensures shuffled and ordered data for distributed training)
    add_rand_idx(goal_len_collection)
    add_rand_idx(goal_proven_task)
    add_rand_idx(rank_collection)
