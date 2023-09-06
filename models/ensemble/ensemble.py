import torch
import torch.nn as nn

'''

Ensemble model, currently composed of a GNN model and a Transformer model. Combines embeddings from both with a 
small MLP which projects back to the originial embedding dimension.  

'''
class EnsembleEmbedder(nn.Module):
    def __init__(self, gnn_model, transformer_model, d_model, global_pool='max', dropout=0):
        super(EnsembleEmbedder, self).__init__()
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model

        self.reduce_proj = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(d_model * 2, d_model),
                                         nn.ReLU())

        self.global_pool = global_pool

    def forward(self, data):
        # assume data is passed as a tuple, formatted in order of which model to apply to
        outs = torch.cat([self.gnn_model(data[0]), self.transformer_model(data[1])], dim=1)
        outs = self.reduce_proj(outs)
        return outs

        # # alternative ensemble could pool/attention across node representations from both encoders:
        # if self.global_pool == 'sum':
        #     return torch.sum(outs, dim=0)
        # elif self.global_pool == 'max':
        #     return torch.max(outs, dim=0)
        # elif self.global_pool == 'mean':
        #     return torch.mean(outs, dim=0)
        # else:
        #     raise NotImplementedError
