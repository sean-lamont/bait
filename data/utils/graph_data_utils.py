"""
Data Transforms and utilities
"""
import torch.nn
from torch_geometric.data import Data, Batch
import torch
import torch_geometric
import logging

"""
DirectedData class, used for batches with attention_edge_index in SAT models
"""


class DirectedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'attention_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


'''
Function to generate a "complete_edge_index" given a ptr corresponding to a PyG batch.
 This is used in vanilla Structure Aware Attention (SAT) models with full attention.
'''


def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


'''
Utility functions for computing ancestor and descendant nodes and node depth for a PyG graph. 
Used for masking attention in Structure Aware Transformer (SAT) Models
'''


def get_directed_edge_index(num_nodes, edge_idx):
    # empty, or too large
    if num_nodes == 1 or edge_idx.shape[1] > 1850:
        return torch.LongTensor([[], []])

    from_idx = []
    to_idx = []

    for i in range(0, num_nodes):
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx)
        except Exception as e:
            logging.warning(f"Exception {e}, {i}, {edge_idx}, {num_nodes}")
            continue

        found_nodes = list(ancestor_nodes.numpy())

        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        try:
            children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx,
                                                                                  flow='target_to_source')
        except Exception as e:
            logging.warning(f"Exception {e}, {i}, {edge_idx}, {num_nodes}")
            continue

        found_nodes = list(children_nodes.numpy())

        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


def get_depth_from_graph(num_nodes, edge_index):
    to_idx = edge_index[1]

    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i,
                                                                          edge_index=edge_index,
                                                                          flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes

    return depths


def to_sequence_batch(data_list, max_len=1024):
    if data_list == []:
        data_list = torch.LongTensor([[0]])

    data_list = torch.nn.utils.rnn.pad_sequence(data_list)
    data_list = data_list[:max_len]
    mask = (data_list == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)
    return (data_list, mask)


def list_to_relation(data_list, max_len):
    xis = [d[0] for d in data_list][:max_len]
    xjs = [d[1] for d in data_list][:max_len]
    edge_attrs = [d[2] for d in data_list][:max_len]

    xi = torch.nn.utils.rnn.pad_sequence(xis)
    xj = torch.nn.utils.rnn.pad_sequence(xjs)
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attrs)

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    return Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)


# return list of tuples, with graph and sequence data in first/second positions (currently only (Graph, Sequence) ensembles)
def to_ensemble_batch(data_list, attributes):
    data_list_0 = [a[0] for a in data_list]
    data_list_1 = [a[1] for a in data_list]
    data_list_0 = to_graph_batch(data_list_0, attributes)
    data_list_1 = to_sequence_batch(data_list_1, attributes['max_len'])
    data_list_1 = (data_list_1[0], data_list_1[1])
    return (data_list_0, data_list_1)


def to_graph_batch(data_list, attributes):
    data_list = Batch.from_data_list(data_list)
    if 'attention_edge' in attributes and attributes['attention_edge'] == 'full':
        data_list.attention_edge_index = ptr_to_complete_edge_index(data_list.ptr)
    return data_list


# todo rename/refactor to 'Transforms'
def transform_expr(expr, data_type, vocab, config=None):
    if data_type == 'graph':
        data = DirectedData(x=torch.LongTensor([vocab[a] if a in vocab else vocab['UNK'] for a in expr['tokens']]),
                            edge_index=torch.LongTensor(expr['edge_index']),
                            edge_attr=torch.LongTensor(expr['edge_attr']), )

        if config is not None:
            if 'attention_edge' in config.attributes and config.attributes['attention_edge'] == 'directed':
                data.attention_edge_index = torch.LongTensor(expr['attention_edge_index'])
            if 'pe' in config.attributes:
                data.abs_pe = torch.LongTensor(expr[config.attributes['pe']])

        return data

    elif data_type == 'sequence':
        return torch.LongTensor([vocab[a] if a in vocab else vocab['UNK'] for a in expr['sequence']])

    elif data_type == 'ensemble':
        return (transform_expr(expr, 'graph', vocab, config), transform_expr(expr, 'sequence', vocab, config))

    elif data_type == 'fixed':
        return expr

    elif data_type == 'relation':
        x = [vocab[a] if a in vocab else vocab['UNK'] for a in expr['tokens']]
        edge_index = expr['edge_index']
        edge_attr = torch.LongTensor(expr['edge_attr'])
        xi = torch.LongTensor([x[i] for i in edge_index[0]])
        xj = torch.LongTensor([x[i] for i in edge_index[1]])
        return (xi, xj, edge_attr)



    # add other transforms here, map from stored expression data to preprocessed format.
    # Could include e.g. RegEx transforms to tokenise on the fly and avoid storing in memory
    # could also include positional encoding computations beyond default, e.g. Magnetic Laplacian for graphs
    # need to specify the transformation in the data_config, data_type,
    # as well as add any additional processing for converting a list of elements into a batch for the model


def transform_batch(batch, config):
    if config.type == 'graph':
        return to_graph_batch(batch, config.attributes)
    elif config.type == 'sequence':
        return to_sequence_batch(batch, config.attributes['max_len'])
    elif config.type == 'relation':
        return list_to_relation(batch, config.attributes['max_len'])
    elif config.type == 'fixed':
        return batch
    elif config.type == 'ensemble':
        return to_ensemble_batch(batch, config.attributes)
