import glob
from data.hol4.mongo_to_torch import get_depth_from_graph, get_directed_edge_index
import torch
from pymongo import MongoClient
import random
import pickle
from ast_def_mizar import goal_to_graph, graph_to_dict
from tqdm import tqdm

if __name__ == '__main__':
    source = 'mongo'
    add_attention = False
    file_dir = 'nnhpdata'

    files = glob.glob(file_dir + '/*')

    expression_dict = {}
    mizar_labels = []

    for file in tqdm(files):
        pos_thms = []
        neg_thms = []

        with open(file) as f:
            lines = f.readlines()

        assert lines[0][0] == 'C'

        for line in lines:
            if line[0] == 'C':
                conj = line[1:].strip("\n")
                if conj not in expression_dict:
                    expression_dict[conj] = graph_to_dict(goal_to_graph(conj))
            elif line[0] == '-':
                neg_thm = line[1:].strip("\n")
                neg_thms.append(neg_thm)
                if neg_thm not in expression_dict:
                    expression_dict[neg_thm] = graph_to_dict(goal_to_graph(neg_thm))
            elif line[0] == '+':
                pos_thm = line[1:].strip("\n")
                pos_thms.append(pos_thm)
                if pos_thm not in expression_dict:
                    expression_dict[pos_thm] = graph_to_dict(goal_to_graph(pos_thm))
            else:
                raise Exception("Not valid")

        mizar_labels.append((conj, pos_thms, neg_thms))

    random.shuffle(mizar_labels)

    train_data = mizar_labels[:int(0.8 * len(mizar_labels))]
    val_data = mizar_labels[
               int(0.8 * len(mizar_labels)):int(0.9 * len(mizar_labels))]

    test_data = mizar_labels[int(0.9 * len(mizar_labels)):]

    train_pairs = []
    for conj, pos_thms, neg_thms in train_data:
        for pos_thm in pos_thms:
            train_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            train_pairs.append((conj, neg_thm, 0))

    val_pairs = []
    for conj, pos_thms, neg_thms in val_data:
        for pos_thm in pos_thms:
            val_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            val_pairs.append((conj, neg_thm, 0))

    test_pairs = []
    for conj, pos_thms, neg_thms in test_data:
        for pos_thm in pos_thms:
            test_pairs.append((conj, pos_thm, 1))
        for neg_thm in neg_thms:
            test_pairs.append((conj, neg_thm, 0))

    vocab = {}
    idx = 0
    for i, k in enumerate(expression_dict.keys()):
        polished_goal = [c for c in k.split(" ") if c != '' and c != '\n']
        expression_dict[k]['full_tokens'] = polished_goal
        for tok in polished_goal:
            if tok not in vocab:
                # reserve 0 for padding idx
                vocab[tok] = idx + 1
                idx += 1

    vocab['VAR'] = len(vocab)
    vocab['VARFUNC'] = len(vocab)

    db = MongoClient()
    db = db['mizar40']
    expression_col = db['expression_graphs']
    vocab_col = db['vocab']
    split_col = db['split_data']

    if add_attention:
        print (f"Adding Attention Edge Index and Graph Data")
        for k, v in tqdm(expression_dict.items()):
            attention_edge_index = get_directed_edge_index(len(v['tokens']),
                                                           torch.LongTensor(v['edge_index'])).tolist()

            depth = get_depth_from_graph(len(v['tokens']), torch.LongTensor(v['edge_index'])).tolist()

            if source == 'mongo':
                    expression_col.insert_one({'_id': k, 'data': {'tokens': v['tokens'],
                                                                  'edge_index': v['edge_index'],
                                                                  'edge_attr': v['edge_attr'],
                                                                  'full_tokens': v['full_tokens'],
                                                                  'attention_edge_index': attention_edge_index,
                                                                  'depth': depth
                                                                  }})
            else:
                v['attention_edge_index'] = attention_edge_index
                v['depth'] = depth

    if source == 'mongo':
        print ("Adding Data to MongoDB")
        for conj, stmt, y in tqdm(train_pairs):
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'y': y, 'split':'train'})

        for conj, stmt, y in tqdm(val_pairs):
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'y': y, 'split':'val'})

        for conj, stmt, y in tqdm(test_pairs):
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'y': y, 'split':'test'})

        for k,v in tqdm(vocab.items()):
            vocab_col.insert_one({'_id': k, 'index': v})

        # if attention, already added
        if not add_attention:
            for k, v in tqdm(expression_dict.items()):
                expression_col.insert_one({'_id': k, 'data': v})

    else:
        with open("mizar_data_new.pk", "wb") as f:
            pickle.dump({'expr_dict': expression_dict, 'train_data': train_pairs, 'val_data': val_pairs,
                         'test_data': test_pairs, 'vocab': vocab}, f)