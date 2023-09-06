import torch
from pymongo import MongoClient
from torch_geometric.data import Data

from experiments.holist.utilities.sexpression_to_graph import sexpression_to_graph
from data.utils.pyg_plot import plot_pyg_graph_with_graphviz

if __name__ == '__main__':
    # client = MongoClient()
    # db = client['holist']
    # expr = db['expression_graphs']
    #
    # cursor = expr.find({}).limit(100)
    # vals = [v for v in cursor]
    # vals = [val if len(val['data']['tokens']) < 20 else None for val in vals]
    # vals = [val for val in vals if val is not None]
    # print (vals)
    #
    sexp = '(a (a (c (fun (bool) (fun (bool) (bool))) =) (a (c (fun (bool) (bool)) ~) (a (c (fun (bool) (bool)) ~) (v (bool) b)))) (v (bool) b))'
    'a a c fun bool fun bool bool = a c fun bool bool ~ a c fun bool bool ~ v bool b v bool b'

    g = sexpression_to_graph(sexp)
    data = Data(torch.arange(len(g['tokens'])), torch.LongTensor(g['edge_index']), torch.LongTensor(g['edge_attr']))
    plot_pyg_graph_with_graphviz(data, g['tokens'])

