import argparse
import re
import logging
import os
import pickle
import sys
import random

# dataset source http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
from pymongo import MongoClient

from data.premise_selection.holstep.data_util import data_loader
from data.premise_selection.holstep.data_util.generate_hol_dataset import generate_dataset
from data.premise_selection.holstep.data_util.holstep_parser import parse_formula, sexpression_from_formula, \
    graph_from_hol_stmt, tree_from_hol_stmt

if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(
        description='Generate graph repr dataset from HolStep')

    parser.add_argument('--path', type=str, help='Path to the root of HolStep dataset', default='data/holstep/raw_data')
    parser.add_argument('--output', type=str, help='Output folder', required=False)
    parser.add_argument(
        '--train_partition',
        '-train',
        type=int,
        help='Number of the partition of the training dataset. Default=200',
        default=200)
    parser.add_argument(
        '--test_partition',
        '--test',
        type=int,
        help='Number of the partition of the testing dataset. Default=20',
        default=20)
    parser.add_argument(
        '--valid_partition',
        '--valid',
        type=int,
        help='Number of the partition of the validation dataset. Default=20',
        default=20)
    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree of graph (default).')
    parser.add_argument(
        '--destination_source',
        type=str,
        default='mongodb',
        help='Where to save data. Either MongoDB or to disk')

    args = parser.parse_args()

    format_choice = {
        'graph': lambda x, y: graph_from_hol_stmt(x, y),
        'tree': lambda x, y: tree_from_hol_stmt(x, y)}

    # assert os.path.isdir(args.output), 'Data path must be a folder'

    assert os.path.isdir(args.path), 'Saving path must be a folder'
    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')
    valid_path = os.path.join(args.path, 'valid')

    files = os.listdir(train_path)
    valid_files = random.sample(files, int(len(files) * 0.07 + 0.5))
    train_files = [x for x in files if x not in valid_files]

    train_expr, train_split = generate_dataset(train_path, args.train_partition,
                                               format_choice[args.format], train_files)

    test_expr, test_split = generate_dataset(test_path, args.test_partition,
                                             format_choice[args.format])

    val_expr, val_split = generate_dataset(train_path, args.valid_partition,
                                           format_choice[args.format], valid_files)

    TOKEN_RE = re.compile(r'[(),]|[^\s(),]+')

    if args.destination_source == 'mongodb':
        client = MongoClient()
        db = client['holstep']
        expr_col = db['expression_graphs']
        split_col = db['split_data']
        vocab_col = db['vocab']

        logging.info("Adding expression Dictionary to MongoDB..")
        train_expr.update(test_expr)
        train_expr.update(val_expr)

        logging.info("Adding vocab Dictionary to MongoDB..")
        loader = data_loader.DataLoader("data/holstep/raw_data/train", "data/holstep/raw_data/hol_train_dict")
        # add dictionary to mongodb

        vocab = {}

        for k, v in loader.dict.items():
            vocab[k] = v + 1
            vocab_col.insert_one({"_id": k, "index": v})

        for k, v in train_expr.items():
            d = {"_id": k, "data": v}

            sexp = parse_formula(k, '')[1]
            seq = sexpression_from_formula(sexp)
            d['data']['sequence'] = seq

            for tok in d['data']['sequence']:
                if d not in vocab:
                    new_ind = len(vocab)
                    vocab[tok] = new_ind
                    vocab_col.insert_one({'id': tok, 'index': new_ind})

            expr_col.insert_one({'_id': d, 'data': d['data']})

        vocab_col.insert_one({"_id": 'UNK', "index": len(vocab) + 1})

    logging.info("Adding training split data to MongoDB..")

    for i, data in enumerate(train_split):
        for flag, conj, stmt in data:
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'train', 'y': flag})

    for i, data in enumerate(val_split):
        for flag, conj, stmt in data:
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'val', 'y': flag})

    for i, data in enumerate(test_split):
        for flag, conj, stmt in data:
            split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'test', 'y': flag})


    else:
        train_output = os.path.join(args.output, 'train')
        test_output = os.path.join(args.output, 'test')
        valid_output = os.path.join(args.output, 'valid')
        if not os.path.exists(train_output):
            os.mkdir(train_output)
        if not os.path.exists(test_output):
            os.mkdir(test_output)
        if not os.path.exists(valid_output):
            os.mkdir(valid_output)

        with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
            pickle.dump(train_expr, f)
        with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
            pickle.dump(val_expr, f)
        with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
            pickle.dump(test_expr, f)

        partition = args.train_partition
        digits = len(str(partition))
        for i, data in enumerate(train_split):
            with open(
                    os.path.join(train_output, 'holstep' + format(i, "0{}d".format(digits))),
                    'wb') as f:
                print('Saving to file {}/{}'.format(i + 1, partition))
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        partition = args.val_partition
        digits = len(str(partition))
        for i, data in enumerate(val_split):
            with open(
                    os.path.join(valid_output, 'holstep' + format(i, "0{}d".format(digits))),
                    'wb') as f:
                print('Saving to file {}/{}'.format(i + 1, partition))
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        partition = args.test_partition
        digits = len(str(partition))
        for i, data in enumerate(test_split):
            with open(
                    os.path.join(test_output, 'holstep' + format(i, "0{}d".format(digits))),
                    'wb') as f:
                print('Saving to file {}/{}'.format(i + 1, partition))
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
