from models.embedding_models.gnn.formula_net.formula_net import FormulaNetSAT
import math
from models.embedding_models.sat.models import GraphTransformer
import torch

from torch import nn


'''

Positional Encoding for Depth Vector as defined in DAGFormer paper

'''


class DepthPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_depth=300, dropout=0.):
        super().__init__()
        depths = torch.arange(max_depth).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_depth, d_model)
        pe[:, 0::2] = torch.sin(depths * div_term)
        pe[:, 1::2] = torch.cos(depths * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = torch.index_select(self.pe, 0, x)
        return self.dropout(x)



'''

MP GNN followed by Transformer with Positional Encoding and Directed Attention Masking

'''


class GNNTransformer(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_dim,
                 num_iterations,
                 dim_feedforward,
                 global_pool='mean',
                 max_edges=200,
                 edge_dim=32,
                 batch_norm=True,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.,
                 abs_pe=False
                 ):

        super(GNNTransformer, self).__init__()

        self.num_iterations = num_iterations

        self.gnn = FormulaNetSAT(embedding_dim=embedding_dim, num_iterations=num_iterations, batch_norm=batch_norm)

        self.initial_encoder = torch.nn.Embedding(input_size, embedding_dim)

        self.edge_encoder = nn.Embedding(max_edges, edge_dim)

        self.transformer_encoder = GraphTransformer(in_size=embedding_dim,
                                                    dim_feedforward=dim_feedforward,
                                                    d_model=embedding_dim,
                                                    num_class=0,
                                                    num_heads=num_heads,
                                                    num_layers=num_layers,
                                                    dropout=dropout,
                                                    in_embed=False,
                                                    use_edge_attr=False,
                                                    global_pool=global_pool,
                                                    batch_norm=batch_norm,
                                                    abs_pe=False,
                                                    # setting k_hop = 0 is equivalent to using a Message Passing Transformer
                                                    k_hop=0,
                                                    )

        if abs_pe:
            self.abs_pe = DepthPositionalEncoding(d_model=embedding_dim, dropout=dropout)
        else:
            self.abs_pe = None

    def forward(self, batch):
        nodes = batch.x
        edges = batch.edge_index
        edge_attr = batch.edge_attr

        nodes = self.initial_encoder(nodes)
        edge_attr = self.edge_encoder(edge_attr)

        nodes = self.gnn(nodes, edges, edge_attr)

        if self.abs_pe:
            nodes = nodes + self.abs_pe(batch.abs_pe)

        batch.x = nodes

        return self.transformer_encoder(batch)