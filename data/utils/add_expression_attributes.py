import os
from multiprocessing import Pool

import torch
from pymongo import MongoClient
from tqdm import tqdm

from data.utils.graph_data_utils import get_directed_edge_index, get_depth_from_graph

# todo take arguments and options

if __name__ == '__main__':

    client = MongoClient()
    db = client['leanstep_sexpression']
    expr_collection = db['expression_graphs']

    def update_attention_func(item):
        expr_collection.update_many({"_id": item["_id"]},
                                    {"$set":
                                        {
                                            "data.attention_edge_index":
                                                get_directed_edge_index(len(item['data']['tokens']),
                                                                        torch.LongTensor(
                                                                            item['data']['edge_index'])).tolist(),
                                            "data.depth":
                                                get_depth_from_graph(len(item['data']['tokens']),
                                                                     torch.LongTensor(
                                                                         item['data']['edge_index'])).tolist()
                                        }})

    items = []
    for item in tqdm(expr_collection.find({})):
        items.append(item)

    pool = Pool(processes=os.cpu_count() // 2)
    for _ in tqdm(pool.imap_unordered(update_attention_func, items), total=len(items)):
        pass


