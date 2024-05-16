import glob
import math
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
    # add traces for each theorem
    # assuming the format path_prefix/{iteration}/trace and each theorem has the same last level filename
    for file in files:
        if file.name not in file_dict:
            file_dict[file.name] = [(file, file.parts[-2])]
        else:
            file_dict[file.name].append((file, file.parts[-2]))

    # todo can augment with e.g. traces from human proofs as in HTPS, or more complex filters, multiple traces per thm etc.

    # simple filter, take most recent trace for unproven, shortest proof for proven
    ret_files = []
    for thm, files in tqdm(file_dict.items()):
        # sort traces by most recent first
        files = sorted(files, key=lambda x: x[1], reverse=True)

        # initialise to most recent
        best_trace = files[0][0]
        best_len = math.inf

        for file, iteration in files:
            with open(file, "rb") as f:
                trace = pickle.load(f)

            if trace.proof and len(trace.proof) < best_len:
                best_trace = file
                best_len = len(trace.proof)

        ret_files.append(best_trace.as_posix())

    return ret_files