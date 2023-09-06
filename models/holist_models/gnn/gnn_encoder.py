import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.utils import degree

'''

 GNN from Paliwal et al.
 
'''


class MLPAggr(nn.Module):
    def __init__(self, embedding_dim, dropout=0.5):
        super().__init__()
        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * embedding_dim, 2 * embedding_dim),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * embedding_dim, embedding_dim),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(embedding_dim, embedding_dim),
                       ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPChildNodes(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__(aggr='sum', flow='target_to_source')

        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * in_channels, 2 * out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * out_channels, out_channels),
                       ReLU(), )
        # nn.Dropout(dropout),
        # Linear(out_channels, out_channels),
        # ReLU())

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


class MLPParentNodes(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__(aggr='sum', flow='source_to_target')

        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * in_channels, 2 * out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * out_channels, out_channels),
                       ReLU(), )
        # nn.Dropout(dropout),
        # Linear(out_channels, out_channels),
        # ReLU())

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        # edge_index[1] gives the number of messages to each node, with degree being the number of parents
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


class GNNEncoder(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, max_edges=3, global_pool=True, dropout=0.5,
                 in_embed=True):
        super().__init__()
        self.num_iterations = num_iterations
        self.global_pool = global_pool

        self.in_embed = in_embed

        if self.in_embed:
            self.initial_encoder = nn.Sequential(nn.Embedding(input_shape, embedding_dim * 2),
                                                 nn.Dropout(dropout),
                                                 nn.Linear(embedding_dim * 2, embedding_dim),
                                                 nn.ReLU(), )
            # nn.Dropout(dropout),
            # nn.Linear(embedding_dim, embedding_dim),
            # nn.ReLU())

            self.edge_encoder = nn.Sequential(nn.Embedding(max_edges, embedding_dim * 2),
                                              nn.Dropout(dropout),
                                              nn.Linear(embedding_dim * 2, embedding_dim),
                                              nn.ReLU(), )
            # nn.Dropout(dropout),
            # nn.Linear(embedding_dim, embedding_dim),
            # nn.ReLU())

        self.mlp_parent_nodes = MLPParentNodes(embedding_dim, embedding_dim, dropout=dropout)

        self.mlp_child_nodes = MLPChildNodes(embedding_dim, embedding_dim, dropout=dropout)

        self.final_agg = MLPAggr(embedding_dim, dropout=dropout)

        # 1x1 conv equivalent to linear projection in output channel
        self.out_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),)

    def forward(self, batch):
        nodes = batch.x
        edges = batch.edge_index
        edge_attr = batch.edge_attr

        if self.in_embed:
            nodes = self.initial_encoder(nodes)
            edge_attr = self.edge_encoder(edge_attr)

        for t in range(self.num_iterations):
            parent_sum = self.mlp_parent_nodes(nodes, edges, edge_attr)
            child_sum = self.mlp_child_nodes(nodes, edges, edge_attr)
            node_update = self.final_agg(torch.cat([nodes, parent_sum, child_sum], dim=-1))
            nodes = nodes + node_update

        if self.global_pool:
            nodes = self.out_proj(nodes)
            return gmp(nodes, batch.batch)
        else:
            return nodes
