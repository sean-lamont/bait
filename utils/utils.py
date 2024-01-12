from __future__ import division, absolute_import, print_function

import torch
from graphviz import Digraph
from omegaconf import OmegaConf
from torch.autograd import Variable

'''

Debugging utility to print out gradient computation graph

'''


def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot


'''
Generator util to iterate through a MongoDB cursor with a given batch size
'''


def get_batches(cursor, batch_size):
    batch = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            yield batch
            del batch[:]
        batch.append(row)
    yield batch


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
