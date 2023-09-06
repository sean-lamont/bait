import random

import numpy as np
from tqdm import tqdm
import json
from pymongo import MongoClient
import pickle


def add_databases():
    data_dir = "data/hol4/data/"

    with open(data_dir + "dep_data.json") as f:
        dep_data = json.load(f)

    with open(data_dir + "paper_goals.pk", "rb") as f:
        paper_dataset = pickle.load(f)

    with open(data_dir + "new_db.json") as f:
        full_db = json.load(f)

    with open(data_dir + "torch_graph_dict.pk", "rb") as f:
        torch_graph_dict = pickle.load(f)

    with open(data_dir + "train_test_data.pk", "rb") as f:
        train_test_data = pickle.load(f)

    with open(data_dir + "graph_token_encoder.pk", "rb") as f:
        token_enc = pickle.load(f)

    new_db = {v[2]: v for k, v in full_db.items()}

    # Issue with multiple expressions with the same polished value in paper goals.
    # Temporary fix to just take the plain paper goal term whenever it appears in new_db
    for goal in paper_dataset:
        if goal[0] in new_db:
            new_db[goal[0]][5] = goal[1]

    with open(data_dir + "adjusted_db.json", "w") as f:
        json.dump(new_db, f)

    # Find goals from original paper which have a database entry
    valid_goals = []
    for goal in paper_dataset:
        if goal[0] in new_db.keys():
            valid_goals.append(goal)

    print(f"Len valid {len(valid_goals)}")
    np.random.shuffle(valid_goals)

    with open(data_dir + "valid_goals_shuffled.pk", "wb") as f:
        pickle.dump(valid_goals, f)

    # Database used for replicating TacticZero, and for pretraining using HOL4 dependency information.
    # Contains information from the HOL4 standard library up to and including "probabilityTheory"
    db_name = "hol4_original_ast"

    # Collection containing meta information about an expression (library, theorem name, etc.)
    info_name = "expression_metadata"

    # Collection containing dependency information for expressions
    dep_name = "dependency_data"

    # Collection containing (goal, premise) pairs for pretraining
    split_name = "split_data"

    # Collection containing the goals from original paper, shuffled
    paper_name = "paper_goals"

    # Collection mapping polished expression to graph representation (one-hot indices, edge index, edge attributes)
    expression_graph_name = "expression_graphs"

    # Collection mapping token to one-hot index
    vocab_name = "vocab"

    db_client = MongoClient()

    db = db_client[db_name]

    dependency_data = db[dep_name]
    pretrain_data = db[split_name]
    paper_split = db[paper_name]
    expression_graph_data = db[expression_graph_name]
    vocab = db[vocab_name]
    expression_info_data = db[info_name]

    print(f"Adding HOL4 standard library data up to and including \"probabilityTheory\" to database {db_name}\n")
    #
    for k, v in tqdm(dep_data.items()):
        info = dependency_data.insert_one(
            {"_id": k,
             "dependencies": v})

    for (k, v) in tqdm(torch_graph_dict.items()):
        info = expression_graph_data.insert_one({"_id": k, "data": v})

    train, val, test, enc_nodes = train_test_data

    for conj, stmt, y in tqdm(train):
        info = pretrain_data.insert_one(
            {"split": "train", "conj": conj, "stmt": stmt, "y": y})

    for conj, stmt, y in tqdm(val):
        info = pretrain_data.insert_one(
            {"split": "val", "conj": conj, "stmt": stmt, "y": y})

    for conj, stmt, y in tqdm(test):
        info = pretrain_data.insert_one(
            {"split": "test", "conj": conj, "stmt": stmt, "y": y})

    # mapping = {i: v for i, v in enumerate(token_enc.categories_[0])}
    vocab_dict = {}
    i = 1
    for v in torch_graph_dict.values():
        toks = v['full_tokens'] + v['tokens']
        for tok in toks:
            if tok not in vocab_dict:
                vocab_dict[tok] = i
                i += 1

    vocab_dict['VAR'] = len(vocab_dict)
    vocab_dict['VARFUNC'] = len(vocab_dict)
    vocab_dict['UNK'] = len(vocab_dict)

    for k, v in tqdm(vocab_dict.items()):
        info = vocab.insert_one({"_id": k, "index": v})

    for k, v in tqdm(new_db.items()):
        info = expression_info_data.insert_one(
            {"_id": k, "theory": v[0], "name": v[1], "dep_id": v[3], "type": v[4], "plain_expression": v[5]})



    random.shuffle(valid_goals)
    train_goals = valid_goals[:int(0.8 * len(valid_goals))]
    val_goals = valid_goals[int(0.8 * len(valid_goals)):]

    info = paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'train'} for g in train_goals])
    info = paper_split.insert_many([{"_id": g[0], "plain": g[1], 'split': 'val'} for g in val_goals])

