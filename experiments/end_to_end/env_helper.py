"""

Helper functions for environment setup used in end-to-end experiments.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle

from data.HOList.utils import io_util
from environments.HOL4.hol4_env import HOL4Env
from environments.HOList.holist_env import HOListEnv
from environments.HOList.proof_assistant import proof_assistant_pb2
from environments.LeanDojo.get_lean_theorems import _get_theorems
# from environments.LeanDojo.leandojo_env import LeanDojoEnv
# from environments.LeanDojo.leandojo_no_split import LeanDojoNoSplitEnv
from environments.LeanDojo.lean4_env import Lean4Env
from experiments.end_to_end.common import zip_strict

# todo hack for now, just load theorem database globally for hol4 get_thm_name
# hol4_thm_db = json.load(open('/home/sean/Documents/phd/bait/data/HOL4/data/adjusted_db.json'))


def get_thm_name(env, thm):
    if env == 'holist':
        return str(thm.fingerprint)
    elif env == 'leandojo' or env == 'leandojo_no_split':
        return str(thm.full_name)
    elif env == 'hol4':
        # theoryName.LemmaName
        return '.'.join(hol4_thm_db[thm[0]][:2])
    else:
        raise NotImplementedError


def get_env(cfg):
    if cfg == 'leandojo':
        return Lean4Env
    # elif cfg == 'leandojo_no_split':
    #     return LeanDojoNoSplitEnv
    elif cfg == 'holist':
        return HOListEnv
    elif cfg == 'hol4':
        return HOL4Env
    else:
        raise NotImplementedError


def get_hol4_theorems(thm_db, goal_db, prev_theorems):
    thm_db = json.load(open(thm_db))

    goals = pickle.load(open(goal_db, "rb"))
    final_theorems = []

    for theorem in goals:
        # for theorem in enumerate(thm_db.keys()):
        if theorem in prev_theorems:
            continue
        else:
            final_theorems.append(theorem)

    theorems = final_theorems
    theorems = list(zip_strict(theorems, [thm_db] * len(theorems)))
    return theorems


def get_holist_theorems(thm_db, prev_theorems):
    theorem_db = io_util.load_theorem_database_from_file(
        str(thm_db))

    # todo filter by config split, library etc.
    theorems = [thm for thm in theorem_db.theorems if thm.tag == proof_assistant_pb2.Theorem.THEOREM]

    # Remove proven theorems if resuming
    final_theorems = []

    for i, theorem in enumerate(theorems):
        if theorem.fingerprint in prev_theorems:
            continue
        else:
            final_theorems.append(theorem)

    theorems = final_theorems
    theorems = list(zip_strict(theorems, [theorem_db] * len(theorems)))

    return theorems


def get_lean_thms(config, prev_theorems):
    repo, theorems, positions = _get_theorems(config)

    # Remove proven theorems if resuming
    final_theorems = []
    final_positions = []

    for i, theorem in enumerate(theorems):
        if theorem.full_name in prev_theorems:
            continue
        else:
            final_theorems.append(theorem)
            final_positions.append(positions[i])

    theorems = final_theorems
    positions = final_positions

    theorems = list(zip_strict([repo] * len(theorems), theorems, positions))

    return theorems


def get_theorems(cfg, prev_theorems):
    if cfg.env == 'leandojo' or cfg.env == 'leandojo_no_split':
        return get_lean_thms(cfg, prev_theorems)
    elif cfg.env == 'holist':
        return get_holist_theorems(cfg.path_theorem_database, prev_theorems)
    elif cfg.env == 'hol4':
        return get_hol4_theorems(cfg.path_theorem_database, cfg.path_goal_database, prev_theorems)
