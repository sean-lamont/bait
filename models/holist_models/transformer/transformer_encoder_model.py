import einops
import torch
from torch import nn
from models.transformer.transformer_encoder_model import TransformerEmbedding

"""

Wrapper for transformer, with final layer projection following HOList GNN models

"""


class TransformerWrapper(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, enc=True, in_embed=False, global_pool=True, small_inner=False,
                 max_len=512):

        super().__init__()

        self.global_pool = global_pool

        self.small_inner = small_inner

        if self.small_inner:
            d_model = d_model // 2

        self.expand_proj = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(d_model, d_model * 4),
                                         nn.ReLU(),
                                         nn.Linear(d_model * 4, d_model * 8),
                                         nn.ReLU())

        self.transformer_embedding = TransformerEmbedding(ntoken=None, d_model=d_model, nhead=nhead, d_hid=d_hid,
                                                          nlayers=nlayers, dropout=dropout, enc=enc,
                                                          global_pool=False, in_embed=in_embed, max_len=max_len
                                                          )

        self.embedding = nn.Sequential(nn.Embedding(ntoken, d_model * 2),
                                       nn.Dropout(dropout),
                                       nn.Linear(d_model * 2, d_model),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(d_model, d_model),
                                       nn.ReLU())

        self.cls_token = nn.Parameter(torch.randn(1, d_model))

    def forward(self, data):
        if type(data) == list or type(data) == tuple:
            x = data[0]
            mask = data[1]
        else:
            x = data.data
            mask = data.mask

        x = self.embedding(x)

        cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=x.shape[1])
        x = torch.cat([x, cls_tokens], dim=0)

        out = self.transformer_embedding(x, mask)

        out = self.expand_proj(out)

        if self.global_pool:
            return torch.max(out, dim=1)[0]

        return out
