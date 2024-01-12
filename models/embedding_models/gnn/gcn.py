import torch_geometric.nn as gnn
from models.embedding_models.sat.layers import StructureExtractor
from torch import nn

class GCNGNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers):
        super().__init__()
        self.gcn = StructureExtractor(embed_dim=embedding_dim, num_layers=num_layers, edge_dim=32)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.edge_embed = nn.Embedding(200, 32)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x = self.embed(x)
        edge_attr = self.edge_embed(edge_attr)

        output = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)

        output = gnn.global_max_pool(output, data.batch)

        return output

class DiGCNGNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers):
        super().__init__()
        self.di_gcn = StructureExtractor(embed_dim=embedding_dim, num_layers=num_layers, edge_dim=32, gnn_type='di_gcn')
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.edge_embed = nn.Embedding(200, 32)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.embed(x)
        edge_attr = self.edge_embed(edge_attr)
        output = self.di_gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        output = gnn.global_mean_pool(output, data.batch)
        return output

