import einops
import torch
from torch import nn
from models.embedding_models.transformer.transformer_encoder_model import TransformerEmbedding

class AttentionRelations(nn.Module):
    def __init__(self, ntoken,
                 embed_dim,
                 edge_dim=200,
                 num_heads=8,
                 dropout=0.,
                 num_layers=4,
                 bias=False,
                 global_pool=True,
                 edge_embed_dim=32,
                 **kwargs):

        super().__init__()

        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.bias = bias

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads

        self.transformer_embedding = TransformerEmbedding(ntoken=None, d_model=embed_dim, nhead=num_heads,
                                                          d_hid=embed_dim, nlayers=num_layers, dropout=dropout,
                                                          enc=False, in_embed=False, global_pool=global_pool)

        # self.edge_embed = torch.nn.Embedding(edge_dim, edge_embed_dim)
        self.edge_embed = torch.nn.Linear(edge_dim, edge_embed_dim)

        if isinstance(ntoken, int):
            self.embedding = torch.nn.Embedding(ntoken, embed_dim)
        elif isinstance(ntoken, nn.Module):
            self.embedding = ntoken
        else:
            raise ValueError("Not implemented!")

        self.scale = head_dim ** -0.5

        # self.r_proj = nn.Linear(embed_dim * 2 + (2 * edge_embed_dim), embed_dim, bias=bias)
        self.r_proj = nn.Linear(embed_dim * 2 + edge_embed_dim, embed_dim, bias=bias)

        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)

        if self.bias:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_k.weight)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                batch,
                return_attn=False):

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        softmax_idx = batch.softmax_idx

        x = self.embedding(x) if hasattr(batch, "node_depth") is None else self.embedding(x, batch.node_depth.view(-1,))

        first = torch.index_select(x, 0, edge_index[0])
        last = torch.index_select(x, 0, edge_index[1])

        if edge_attr is not None:
            edge_attr = self.edge_embed(edge_attr)

            # if multiple edge attributes, flatten them
            # if len(edge_attr.shape) > 2:
            #     edge_attr = edge_attr.flatten(1, 2)

            R = torch.cat([first, edge_attr, last], dim=1)
        else:
            R = torch.cat([first, last], dim=1)

        R = self.r_proj(R)

        # cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=len(softmax_idx)- 1)
        cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=len(softmax_idx))

        # split R according to softmax_idx (i.e. how many edges per sequence in batch)
        R = torch.tensor_split(R, softmax_idx[:-1])

        R = torch.nn.utils.rnn.pad_sequence(R)

        R = torch.cat([R, cls_tokens], dim=0)

        enc = self.transformer_embedding(R)

        if self.global_pool:
            return enc

        # get masked inds
        # masked_inds = batch.mask_idx

        # raw_mask = batch.mask_raw
        #
        # get corresponding relations for nodes in mask
        # relation_mask = []
        # mask to get back what the tokens are
        # rev_mask = []
        # for i, ind in enumerate(masked_inds):
        #     if ind in edge_index[0]:
        #         # which relation corresponds to the mask idx
        #         rel_inds = (edge_index[0] == i).nonzero().flatten()
        #         relation_mask.extend(rel_inds)
        #         rev_mask.extend([i] * len(rel_inds))
        #
        #     # elif i in edge_index[1]:
        #     #     rel_inds = (edge_index[1] == i).nonzero().flatten()
        #     #     relation_mask.extend(rel_inds)
        #     #     rev_mask.extend([i] * len(rel_inds))
        #
        # relation_mask_nodes = torch.index_select(enc,
        #
        # print (masked_inds, enc.shape)
        #
        # masked_pos = masked_inds[:, :, None].expand(-1, -1, enc.size(-1))  # [batch_size, max_pred, d_model]

        return
