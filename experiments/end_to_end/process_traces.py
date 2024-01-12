import glob
import pickle

from tqdm import tqdm
from pathlib import *


def get_traces(path):
    files = glob.glob(path, recursive=True)

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


def filter_traces(trace_dir):
    path = Path(trace_dir)

    files = [x for x in path.rglob("*") if x.is_file()]

    file_dict = {}
    # add traces for each theorem, assuming the format path_prefix/{iteration}/trace
    for file in files:
        if file.name not in file_dict:
            file_dict[file.name] = [(file, file.parts[-2])]
        else:
            file_dict[file.name].append((file, file.parts[-2]))

    # todo can add e.g. traces from human proofs as in HTPS, or more complex filters, multiple traces per thm etc.

    # simple filter, take most recent trace for unproven, most recently proven trace for proven
    ret_files = []
    for thm, files in file_dict.items():
        # sort traces by most recent first
        files = sorted(files, key=lambda x: x[1], reverse=True)

        # initialise to most recent
        best_trace = files[0][0]

        for file, iteration in files:
            with open(file, "rb") as f:
                trace = pickle.load(f)

            if trace.proof:
                best_trace = file
                break

        ret_files.append(best_trace.as_posix())

    return ret_files

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
