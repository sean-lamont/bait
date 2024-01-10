import glob
import pickle

from tqdm import tqdm


def get_traces(path):
    files = glob.glob(path)

    traces = []
    for file in tqdm(files):
        with open(file, "rb") as f:
            trace = pickle.load(f)
            traces.append(trace)
    return traces


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

# todo
# def transform_subgoal_proven(subgoal_data, visit_threshold=1024):
#     proof_len = subgoal_data['distance_to_proof']
#     if proof_len < math.inf:
#         return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 1}
#     elif subgoal_data['visits'] >= visit_threshold:
#         return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 0}
#     else:
#         return None
#
#
# # create labelled goal provability data with smooth labels for unproven goals, getting closer to 0 as more visits
# def transform_goal_smooth(subgoal_data, max_count=4096, min_count=256):
#     proof_len = subgoal_data['distance_to_proof']
#     if proof_len < math.inf:
#         return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'], 'target': 1}
#     elif subgoal_data['visits'] >= min_count:
#         return {'initial_goal': subgoal_data['initial_goal'], 'subgoal': subgoal_data['subgoal'],
#                 'target': 0.5 * max(0, (1 - (subgoal_data['visits'] / max_count)))}
#     else:
#         return None
#
#