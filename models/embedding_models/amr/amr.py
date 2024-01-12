import einops
import torch
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class AMREncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", edge_dim=0, layer_norm=True, pre_norm=False,
                 **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = AttentionAMR(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                      bias=False, edge_dim=edge_dim, **kwargs)

        self.layer_norm = layer_norm
        self.pre_norm = pre_norm

        if layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, xs, xt, edge_index, edge_index_source, edge_index_target, softmax_idx,
                edge_attr=None, ptr=None,
                return_attn=False,
                ):
        xs, xt = self.self_attn(
            x_source=xs,
            x_target=xt,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_index_source=edge_index_source,
            edge_index_target=edge_index_target,
            softmax_idx=softmax_idx,
            ptr=ptr,
            return_attn=return_attn
        )

        return xs, xt


class AttentionAMR(gnn.MessagePassing):
    """Multi-head AMR attention implementation using PyG interface

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    """

    def __init__(self, embed_dim, edge_dim=0, num_heads=8, dropout=0., bias=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')

        self.embed_dim = embed_dim
        self.bias = bias

        head_dim = embed_dim // num_heads

        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads

        self.scale = head_dim ** -0.5

        self.r_proj = nn.Linear(embed_dim * 2 + edge_dim, embed_dim, bias=bias)

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.combine_source_target = nn.Linear(embed_dim * 2, embed_dim, bias=bias)

        self.ffn = torch.nn.Sequential(nn.Linear(embed_dim, embed_dim * 4, bias=bias),
                                       nn.ReLU(),
                                       nn.Linear(embed_dim * 4, embed_dim * 2))

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

        # print (f"Attn network {self}")

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)

        if self.bias:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_k.weight)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x_source,
                x_target,
                edge_index,
                edge_index_source,
                edge_index_target,
                softmax_idx,
                edge_attr=None,
                ptr=None,
                return_attn=False):
        """
        Compute attention layer.

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        complete_edge_index:        edge index from fully connected graph
        subgraph_node_index:        documents the node index in the k-hop subgraphs
        subgraph_edge_index:        edge index of the extracted subgraphs
        subgraph_indicator_index:   indices to indicate to which subgraph corresponds to which node
        subgraph_edge_attr:         edge attributes of the extracted k-hop subgraphs
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix

        # print (x_source.shape)
        # print (x_target.shape)
        first = torch.index_select(x_source, 0, edge_index[0])
        last = torch.index_select(x_target, 0, edge_index[1])

        if edge_attr is not None:
            R = torch.cat([first, edge_attr, last], dim=1)
        else:
            R = torch.cat([first, last], dim=1)

        # "complete_edge_index" which has "from" relations "to" source nodes, and "from" relations to the corresponding "target" nodes
        Q_source = self.to_q(x_source)
        Q_target = self.to_q(x_target)

        R = self.r_proj(R)

        V = self.to_v(R)
        K = self.to_k(R)

        attn = None

        out_source = self.propagate(edge_index_source, v=V, qk=(K, Q_source), edge_attr=None, size=None,
                                    return_attn=return_attn, softmax_idx=softmax_idx)

        out_target = self.propagate(edge_index_target, v=V, qk=(K, Q_target), edge_attr=None, size=None,
                                    return_attn=return_attn, softmax_idx=softmax_idx)

        out_source = rearrange(out_source, 'n h d -> n (h d)')

        out_target = rearrange(out_target, 'n h d -> n (h d)')

        out_source = self.out_proj(out_source)

        out_target = self.out_proj(out_target)

        scale = F.sigmoid(self.combine_source_target(torch.cat([out_source, out_target], dim=1)))

        out = scale * out_source + (1 - scale) * out_target

        O_source, O_target = self.ffn(out).chunk(2, dim=-1)

        x_source = self.layer_norm(x_source + O_source)
        x_target = self.layer_norm(x_target + O_target)

        # if return_attn:
        #     attn = self._attn
        #     self._attn = None
        #     attn = torch.sparse_coo_tensor(
        #         complete_edge_index,
        #         attn,
        #     ).to_dense().transpose(0, 1)

        return x_source, x_target

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn, softmax_idx):

        """Self-attention operation compute the dot-product attention """

        # print (f"v_j {v_j}, qk_j: {qk_j}, qk_i: {qk_i}")

        # print (f"index {index}\n\n")

        # todo AMR make sure size_i isn't breaking softmax for non-complete index

        # size_i = max(index) + 1 # from torch_geometric docs? todo test correct

        # qk_j is keys i.e. message "from" j, qk_i maps to queries i.e. messages "to" i

        # index maps to the "to"/ i values i.e. index[i] = 3 means i = 3, and len(index) is the number of messages
        # i.e. index will be 0,n repeating n times (if complete_edge_index is every combination of nodes)

        # print (f"qkj: {qk_j}, qki: {qk_i}, vj: {v_j}")

        # print (f"message: v_j {v_j.shape}, qk_j: {qk_j.shape}, index: {index}, ptr: {ptr}, size_i: {size_i}")

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)

        # print (f"message after: v_j {v_j.shape}, qk_j: {qk_j.shape}, index: {index}, ptr: {ptr}, size_i: {size_i}")

        # sum over dimension, giving n h shape
        attn = (qk_i * qk_j).sum(-1) * self.scale

        # print (attn.shape)

        if edge_attr is not None:
            attn = attn + edge_attr

        # index gives what to softmax over

        # print (f"attn: {attn}, index {index}, soft_ind {softmax_idx}, size {softmax_idx[-1]}, lenidx {len(index)}")
        attn = utils.softmax(attn, ptr=softmax_idx, num_nodes=softmax_idx[-1])
        # print (f"attn after {attn}")
        if return_attn:
            self._attn = attn

        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


class MPAttentionAggr(gnn.MessagePassing):

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
        embed_dim = x.size(1)

        cls_token = nn.Parameter(torch.randn(1, embed_dim))

        bsz = len(data.ptr) - 1

        new_index = torch.vstack((torch.arange(data.num_nodes).to(x), data.batch + data.num_nodes))
        new_index2 = torch.vstack((new_index[1], new_index[0]))
        idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
        new_index3 = torch.vstack((idx_tmp, idx_tmp))

        complete_edge_index = torch.cat((
            new_index, new_index2, new_index3), dim=-1)

        cls_tokens = einops.repeat(cls_token, '() d -> b d', b=bsz).to(x)

        x = torch.cat((x, cls_tokens))

        qk = self.to_qk(x).chunk(2, dim=-1)

        v = self.to_v(x)

        attn = None

        out = self.propagate(complete_edge_index.long(), v=v, qk=qk, edge_attr=None, size=None,
                             return_attn=return_attn)

        out = rearrange(out, 'n h d -> n (h d)')

        if return_attn:
            attn = self._attn
            self._attn = None
            attn = torch.sparse_coo_tensor(
                complete_edge_index,
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


class AMREncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, edge_index_source, edge_index_target, softmax_idx,
                edge_attr=None,
                ptr=None, return_attn=False):
        xs, xt = x, x

        for mod in self.layers:
            xs, xt = mod(xs, xt, edge_index, edge_index_source, edge_index_target,
                         softmax_idx=softmax_idx,
                         edge_attr=edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )

        # if self.norm is not None:
        #     output = self.norm(output)

        return torch.cat([xs, xt], dim=1)


class AMRTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=4,
                 dim_feedforward=512, dropout=0.2, num_layers=2,
                 layer_norm=False, abs_pe=False, abs_pe_dim=0,
                 use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True,
                 global_pool='mean', device='cuda', **kwargs):

        super().__init__()

        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)
        # print ("r_inductunning")
        self.abs_pe = abs_pe

        self.abs_pe_dim = abs_pe_dim

        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Embedding(abs_pe_dim, d_model)
            # self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)

        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        self.device = device

        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = 0

        encoder_layer = AMREncoderLayer(
            d_model, num_heads, dim_feedforward,
            dropout, layer_norm=layer_norm,
            edge_dim=edge_dim, **kwargs)

        self.encoder = AMREncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model * 2))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.attn_pool = MPAttentionAggr(embed_dim=d_model * 2, num_heads=num_heads, dropout=dropout)

    def forward(self, data, return_attn=False):

        x, edge_index, edge_attr, softmax_idx = data.x, data.edge_index, data.edge_attr, data.softmax_idx

        range_ids = torch.arange(edge_index.shape[1]).to(self.device)

        edge_index_source = torch.stack([range_ids, edge_index[0]], dim=0)
        edge_index_target = torch.stack([range_ids, edge_index[1]], dim=0)

        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None

        output = self.embedding(x)

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None

        output = self.encoder(
            x=output,
            edge_index=edge_index,
            edge_index_source=edge_index_source,
            edge_index_target=edge_index_target,
            softmax_idx=softmax_idx,
            edge_attr=edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        # if self.global_pool == 'cls' and self.use_global_pool:
        #     bsz = len(data.ptr) - 1
        #
        #     # complete_edge_index = ptr_to_complete_edge_index(data.ptr.cpu()).cuda()
        #
        #     new_index = torch.vstack((torch.arange(data.num_nodes).to(x), data.batch + data.num_nodes))
        #     new_index2 = torch.vstack((new_index[1], new_index[0]))
        #     idx_tmp = torch.arange(data.num_nodes, data.num_nodes + len(data.ptr) - 1).to(data.batch)
        #     new_index3 = torch.vstack((idx_tmp, idx_tmp))
        #
        #     # complete_edge_index = torch.cat((
        #     #     complete_edge_index, new_index, new_index2, new_index3), dim=-1)
        #
        #     complete_edge_index = torch.cat((
        #          new_index, new_index2, new_index3), dim=-1)
        #
        #     cls_tokens = repeat(self.cls_token, '() d -> b d', b=len(data.ptr) - 1)
        #
        #     output = torch.cat((output, cls_tokens))

        if self.use_global_pool:
            if self.global_pool == 'cls':
                bsz = len(data.ptr) - 1
                output, attn = self.attn_pool(
                    data=Data(x=output, edge_index=data.edge_index, ptr=data.ptr, batch=data.batch,
                              num_nodes=data.num_nodes), return_attn=True)
                output = output[-bsz:]
            else:
                output = gnn.global_max_pool(output, data.batch)

        return output
