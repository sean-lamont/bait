import einops
import torch
from torch import nn
from models.embedding_models.transformer.transformer_encoder_model import TransformerEmbedding

class AttentionRelationSmall(nn.Module):
    def __init__(self, ntoken,
                 embed_dim,
                 edge_dim=200,
                 num_heads=8,
                 dropout=0.,
                 num_layers=4,
                 bias=False,
                 global_pool=True,
                 edge_embed_dim=64,
                 pad=True,
                 inner_dim=None,
                 **kwargs):

        super().__init__()

        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.bias = bias
        self.pad = pad

        self.num_heads = num_heads



        if inner_dim is not None:
            self.inner_dim = inner_dim
        else:
            self.inner_dim = embed_dim // 2

        head_dim = self.inner_dim // num_heads

        assert head_dim * num_heads == self.inner_dim, "embed_dim must be divisible by num_heads"

        self.transformer_embedding = TransformerEmbedding(ntoken=None, d_model=self.inner_dim, nhead=num_heads,
                                                          d_hid=embed_dim, nlayers=num_layers, dropout=dropout,
                                                          enc=False, in_embed=False, global_pool=False)

        if self.pad:
            self.edge_embed = torch.nn.Embedding(edge_dim + 1, edge_embed_dim, padding_idx=0)
        else:
            self.edge_embed = torch.nn.Embedding(edge_dim, edge_embed_dim)
        # self.edge_embed = torch.nn.Linear(edge_dim, edge_embed_dim)

        if isinstance(ntoken, int):
            self.embedding = torch.nn.Embedding(ntoken + 1, self.inner_dim, padding_idx=0)
            # self.embed_2 = torch.nn.Embedding(ntoken + 1, embed_dim, padding_idx=0)
        elif isinstance(ntoken, nn.Module):
            self.embedding = ntoken
        else:
            raise ValueError("Not implemented!")

        self.scale = head_dim ** -0.5


        self.r_proj = nn.Linear(self.inner_dim * 2 + edge_embed_dim, self.inner_dim, bias=bias)

        # self.r_proj = nn.Linear(embed_dim + edge_embed_dim, embed_dim, bias=bias)

        self.in_proj = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), nn.GELU())
        self.out_proj = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), nn.GELU())

        # self.r_proj = nn.Sequential(nn.Linear(embed_dim * 2 + edge_embed_dim, embed_dim, bias=bias),
        #                             nn.ReLU(),
        #                             nn.LayerNorm(embed_dim))


        self.cls_token = nn.Parameter(torch.randn(1, self.inner_dim))


        self.expand_proj = nn.Sequential(nn.Linear(self.inner_dim, embed_dim), nn.GELU())

    def forward(self,
                data,
                return_attn=False):

        xi = data.xi
        xj = data.xj
        edge_attr = data.edge_attr_

        xi = self.embedding(xi)
        xj = self.embedding(xj)

        xi = self.in_proj(xi)
        xj = self.out_proj(xj)

        if edge_attr is not None:
            edge_attr = self.edge_embed(edge_attr)

            # if multiple edge attributes, flatten them
            if len(edge_attr.shape) > 3:
                edge_attr = edge_attr.flatten(-2, -1)

            R = torch.cat([xi, edge_attr, xj], dim=-1)
        else:
            R = torch.cat([xi, xj], dim=-1)

        R = self.r_proj(R)

        cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=R.shape[1])

        R = torch.cat([R, cls_tokens], dim=0)

        enc = self.transformer_embedding(R, mask=data.mask)

        if self.global_pool:
            enc = self.expand_proj(enc)
            # max pool, shape (N , K, D)
            return torch.max(enc, dim=1)[0]
