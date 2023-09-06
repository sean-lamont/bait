import torch
import math
import einops
from einops import rearrange
import torch_geometric.utils as utils
from torch import nn
import torch_geometric.nn as gnn

'''
Implementation of standard transformer through message passing. Generates a fully connected graph on input sequence
and performs self attention using message passing.

Batching is done through PyG, with a batch consisting only of (batch_size, d_model),
as opposed to standard (batch_size, max_seq_len, d_model)
'''


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


class MPAttention(gnn.MessagePassing):

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')

        self.embed_dim = embed_dim
        self.bias = bias

        head_dim = embed_dim // num_heads

        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads

        self.scale = head_dim ** -0.5

        self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)

        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_qk.weight)

        if self.bias:
            nn.init.xavier_uniform_(self.to_qk.weight)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                data,
                return_attn=False):

        x = data.x

        if data.edge_index is None:
            edge_index = ptr_to_complete_edge_index(data.ptr)
        else:
            edge_index = data.edge_index

        x = data.x

        qk = self.to_qk(x).chunk(2, dim=-1)

        v = self.to_v(x)

        attn = None

        out = self.propagate(edge_index, v=v, qk=qk, edge_attr=None, size=None,
                             return_attn=return_attn)

        out = rearrange(out, 'n h d -> n (h d)')

        if return_attn:
            attn = self._attn
            self._attn = None
            attn = torch.sparse_coo_tensor(
                edge_index,
                attn,
            ).to_dense().transpose(0, 1)

        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)

        attn = (qk_i * qk_j).sum(-1) * self.scale

        attn = utils.softmax(attn, index, ptr, size_i)

        if return_attn:
            self._attn = attn

        attn = self.attn_dropout(attn)

        msg = v_j * attn.unsqueeze(-1)

        return msg


class MPTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 **kwargs):

        # print (nhead)
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = MPAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                     bias=False, **kwargs)

        self.batch_norm = batch_norm
        self.pre_norm = pre_norm

        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, complete_edge_index,
                ptr=None,
                return_attn=False,
                ):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            complete_edge_index,
            ptr=ptr,
            return_attn=return_attn
        )

        x = x + self.dropout1(x2)

        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)

        return x


class MPPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, ptr):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        pe_ptr = torch.cat([self.pe[:(ptr[i + 1] - ptr[i])] for i in range(len(ptr) - 1)], dim=0)

        # print (pe_ptr.shape)

        return pe_ptr


class MPTransformerEncoder(nn.TransformerEncoder):

    def forward(self, x, edge_index, complete_edge_index, edge_attr=None, ptr=None, return_attn=False):

        output = x

        for mod in self.layers:
            output = mod(output,
                         edge_index=edge_index,
                         complete_edge_index=complete_edge_index,
                         ptr=ptr,
                         return_attn=return_attn
                         )

        if self.norm is not None:
            output = self.norm(output)

        return output


class MPTransformer(nn.Module):

    def __init__(self, in_size, d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 batch_norm=False, pe=False,
                 in_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):

        super().__init__()

        self.pe = pe

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        encoder_layer = MPTransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_norm=batch_norm,
            **kwargs)

        self.encoder = MPTransformerEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool

        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool

        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool

        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None

        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len

    def forward(self, data, return_attn=False):

        output = data.x

        ptr = data.ptr

        complete_edge_index = data.complete_edge_index

        output = self.embedding(output)

        if self.pe:
            # print (data.pe[0], output[0], data.pe.shape, output.shape)
            output = output + data.pe

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1

            # if edge_index is not None:
            #     new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
            #     new_index2 = torch.vstack((new_index[1], new_index[0]))
            #     idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
            #     new_index3 = torch.vstack((idx_tmp, idx_tmp))
            #     edge_index = torch.cat((
            #         edge_index, new_index, new_index2, new_index3), dim=-1)

            degree = None

            cls_tokens = einops.repeat(self.cls_token, '() d -> b d', b=bsz)

            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            edge_index=None,
            complete_edge_index=complete_edge_index,
            ptr=data.ptr,
            return_attn=return_attn
        )

        if self.use_global_pool:

            if self.global_pool == 'cls':
                output = output[-bsz:]

            else:
                # output_1 = self.pooling(output, data.batch)
                output = gnn.global_max_pool(output, data.batch)
                # output = torch.cat([output_1, output_2], dim=1)

        return output
