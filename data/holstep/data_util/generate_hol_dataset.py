import argparse
from tqdm import tqdm
from pymongo import MongoClient
import logging
import os
import pickle
import random
import sys
sys.path.insert(0, 'data_util')
import torch

from data.holstep.data_util import data_loader
from data.holstep.data_util.holstep_parser import graph_from_hol_stmt
from data.holstep.data_util.holstep_parser import tree_from_hol_stmt


def count_stmt(path):
    '''Count the number of the statements in the files of the given folder

    Parameters
    ----------
    path : str
        Path to the directory

    Returns
    -------
    int
        The number of total conjectures and statements
    '''
    total = 0
    files = os.listdir(path)
    for i, fname in enumerate(files):
        fpath = os.path.join(path, fname)
        print('Counting file {}/{} at {}.'.format(i + 1, len(files), fpath))
        with open(fpath, 'r') as f:
            total += sum([1 if line and line[0] in '+-C' else 0 for line in f])
    return total


def generate_dataset(path,  partition, converter, files=None):
    '''Generate dataset at given path

    Parameters
    ----------
    path : str
        Path to the source
    partition : int
        Number of the partition for this dataset (i.e. # of files)
    '''
    outputs = [[] for _ in range(partition)]
    if files is None:
        files = os.listdir(path)

    # print(os.curdir)
    loader = data_loader.DataLoader("data/holstep/raw_data/train", "data/holstep/raw_data/hol_train_dict")


    reverse_vocab = {v: k for k, v in loader.dict.items()}

    expressions = {}

    logging.info("Processing raw HOLStep data..")
    for i, fname in tqdm(enumerate(files)):
        fpath = os.path.join(path, fname)
        # print('Processing file {}/{} at {}.'.format(i + 1, len(files), fpath))
        with open(fpath, 'r') as f:
            next(f)
            conj_symbol = next(f)
            conj_token = next(f)
            assert conj_symbol[0] == 'C'

            if conj_symbol[2:] not in expressions:

                conjecture = converter(conj_symbol[2:], conj_token[2:])

                onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(
                    conjecture)

                edge_index = torch.stack([oindex1, oindex2], dim=0).long()

                conj_graph = {"tokens": [reverse_vocab[c.item()] for c in onehot],
                              "edge_index": edge_index.tolist(),
                              "edge_attr": edge_attr}

                expressions[conj_symbol[2:]] = conj_graph

            for line in f:
                if line and line[0] in '+-':

                    stmt_symbol = line[2:]
                    stmt_token = next(f)[2:]

                    if stmt_symbol not in expressions:
                        statement = converter(stmt_symbol, stmt_token)

                        onehot, iindex1, iindex2, imat, oindex1, oindex2, imat2, edge_attr = loader.directed_generate_one_sentence(
                            statement)

                        edge_index = torch.stack([oindex1, oindex2], dim=0).long()

                        stmt_graph = {"tokens": [reverse_vocab[c.item()] for c in onehot],
                                      "edge_index": edge_index.tolist(),
                                      "edge_attr": edge_attr}

                        expressions[stmt_symbol] = stmt_graph

                    flag = 1 if line[0] == '+' else 0
                    record = flag, conj_symbol[2:], stmt_symbol  # conjecture, statement
                    outputs[random.randint(0, partition - 1)].append(record)

    return expressions, outputs


# if __name__ == '__main__':
#     sys.setrecursionlimit(10000)
#     parser = argparse.ArgumentParser(
#         description='Generate graph repr dataset from HolStep')
#
#     parser.add_argument('path', type=str, help='Path to the root of HolStep dataset')
#     parser.add_argument('output', type=str, help='Output folder', required=False)
#     parser.add_argument(
#         '--train_partition',
#         '-train',
#         type=int,
#         help='Number of the partition of the training dataset. Default=200',
#         default=200)
#     parser.add_argument(
#         '--test_partition',
#         '--test',
#         type=int,
#         help='Number of the partition of the testing dataset. Default=20',
#         default=20)
#     parser.add_argument(
#         '--valid_partition',
#         '--valid',
#         type=int,
#         help='Number of the partition of the validation dataset. Default=20',
#         default=20)
#     parser.add_argument(
#         '--format',
#         type=str,
#         default='graph',
#         help='Format of the representation. Either tree of graph (default).')
#     parser.add_argument(
#         '--destination_source',
#         type=str,
#         default='mongodb',
#         help='Where to save data. Either MongoDB or to disk')
#
#     args = parser.parse_args()
#
#     format_choice = {
#         'graph': lambda x, y: graph_from_hol_stmt(x, y),
#         'tree': lambda x, y: tree_from_hol_stmt(x, y)}
#
#     assert os.path.isdir(args.output), 'Data path must be a folder'
#     assert os.path.isdir(args.path), 'Saving path must be a folder'
#     train_output = os.path.join(args.output, 'train')
#     test_output = os.path.join(args.output, 'test')
#     valid_output = os.path.join(args.output, 'valid')
#     train_path = os.path.join(args.path, 'train')
#     test_path = os.path.join(args.path, 'test')
#     valid_path = os.path.join(args.path, 'valid')
#
#     files = os.listdir(train_path)
#     valid_files = random.sample(files, int(len(files) * 0.07 + 0.5))
#     train_files = [x for x in files if x not in valid_files]
#     # print(valid_files)
#     # print(train_files)
#
#     train_expr, train_split = generate_dataset(train_path,  args.train_partition,
#                                                format_choice[args.format], train_files)
#
#     test_expr, test_split = generate_dataset(test_path,  args.test_partition,
#                                              format_choice[args.format])
#
#     val_expr, val_split = generate_dataset(train_path,  args.valid_partition,
#                                             format_choice[args.format], valid_files)
#
#     if args.destination_source == 'mongodb':
#         client = MongoClient
#         db = client['holstep']
#         expr_col = db['expression_graphs']
#         split_col = db['split_data']
#         vocab_col = db['vocab']
#
#         logging.info("Adding expression Dictionary to MongoDB..")
#         train_expr.update(test_expr)
#         train_expr.update(val_expr)
#
#         for k, v in train_expr.items():
#             expr_col.insert_one({"_id": k, "data": v})
#
#         logging.info("Adding vocab Dictionary to MongoDB..")
#         loader = data_loader.DataLoader("data/holstep/raw_data/train", "data/holstep/raw_data/hol_train_dict")
#         # add dictionary to mongodb
#         vocab_col = []
#         for k, v in loader.dict.items():
#             vocab_col.insert_one({"_id": k, "index": v})
#
#         logging.info("Adding training split data to MongoDB..")
#         for i, data in enumerate(train_split):
#             for flag, conj, stmt in data:
#                 split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'train', 'y': flag})
#
#         for i, data in enumerate(val_split):
#             for flag, conj, stmt in data:
#                 split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'val', 'y': flag})
#
#         for i, data in enumerate(test_split):
#             for flag, conj, stmt in data:
#                 split_col.insert_one({'conj': conj, 'stmt': stmt, 'split': 'test', 'y': flag})
#
#
#     else:
#         if not os.path.exists(train_output):
#             os.mkdir(train_output)
#         if not os.path.exists(test_output):
#             os.mkdir(test_output)
#         if not os.path.exists(valid_output):
#             os.mkdir(valid_output)
#
#         with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
#             pickle.dump(train_expr, f)
#         with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
#             pickle.dump(val_expr, f)
#         with open(os.path.join(train_output, 'expr_dict'), 'wb') as f:
#             pickle.dump(test_expr, f)
#
#         partition = args.train_partition
#         digits = len(str(partition))
#         for i, data in enumerate(train_split):
#             with open(
#                     os.path.join(train_output, 'holstep' + format(i, "0{}d".format(digits))),
#                     'wb') as f:
#                 print('Saving to file {}/{}'.format(i + 1, partition))
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#
#
#         partition = args.val_partition
#         digits = len(str(partition))
#         for i, data in enumerate(val_split):
#             with open(
#                     os.path.join(valid_output, 'holstep' + format(i, "0{}d".format(digits))),
#                     'wb') as f:
#                 print('Saving to file {}/{}'.format(i + 1, partition))
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#
#         partition = args.test_partition
#         digits = len(str(partition))
#         for i, data in enumerate(test_split):
#             with open(
#                     os.path.join(test_output, 'holstep' + format(i, "0{}d".format(digits))),
#                     'wb') as f:
#                 print('Saving to file {}/{}'.format(i + 1, partition))
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
