from typing import Optional
import einops
import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh

'''

Code for Magnetic Laplacian (fork from torch_geometric_signed_directed.utils.directed.get_magnetic_Laplacian)

'''


def get_magnetic_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                           normalization: Optional[str] = 'sym',
                           dtype: Optional[int] = None,
                           num_nodes: Optional[int] = None,
                           q: Optional[float] = 0.25,
                           return_eig: bool = True,
                           k=27):
    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes, "add")

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym / 2

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_sym^{-1/2} A_sym D_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * \
                      edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_eig:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        k_ = min(k - 2, L.shape[0] - 2)
        eig_vals, eig_vecs = eigsh(L, k=k_, which='LM', return_eigenvectors=True)
        eig_vals = torch.FloatTensor(eig_vals)
        eig_real = torch.FloatTensor(eig_vecs.real)
        eig_imag = torch.FloatTensor(eig_vecs.imag)

        if k_ < k:
            eig_real = torch.nn.functional.pad(eig_real, (0, k - 2 - k_), value=0)
            eig_vals = torch.nn.functional.pad(eig_vals, (0, k - 2 - k_), value=0)
            eig_imag = torch.nn.functional.pad(eig_imag, (0, k - 2 - k_), value=0)

        # shape n,k for vecs, shape k for vals
        return eig_vals, (eig_real, eig_imag)


'''

Code for MagLapNet in Torch

'''


class MagLapNet(torch.nn.Module):
    def __init__(self,
                 eig_dim: int = 32,
                 d_embed: int = 256,
                 num_heads: int = 4,
                 n_layers: int = 1,
                 dropout_p: float = 0.2,
                 # return_real_output: bool = True,
                 consider_im_part: bool = True,
                 use_signnet: bool = True,
                 use_attention: bool = False,
                 concatenate_eigenvalues: bool = False,
                 norm=True,
                 ):

        super().__init__()

        self.concatenate_eigenvalues = concatenate_eigenvalues
        self.consider_im_part = consider_im_part
        self.use_signnet = use_signnet
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.element_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, eig_dim) if self.consider_im_part else torch.nn.Linear(1, eig_dim),
            torch.nn.ReLU()
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=eig_dim, nhead=num_heads, dropout=dropout_p)
        self.PE_Transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if norm:
            self.norm = torch.nn.LayerNorm(eig_dim + 1)
        else:
            self.norm = None

        self.proj = torch.nn.Linear(eig_dim, d_embed)

    def forward(self, eigenvalues,
                eigenvectors):

        trans_eig = eigenvectors[0]
        trans_eig_im = eigenvectors[1]

        mask = (trans_eig == 0)

        if self.consider_im_part:
            trans_eig = torch.stack([trans_eig, trans_eig_im], dim=-1)

        trans = self.element_mlp(trans_eig)
        if self.use_signnet:
            trans = trans + self.element_mlp(-trans_eig)

        eigenvalues = einops.repeat(eigenvalues, "k -> k 1")

        if self.concatenate_eigenvalues:
            eigenvalues_ = einops.repeat(eigenvalues, "k 1-> n k 1", n=trans.shape[0])
            trans = torch.cat([eigenvalues_, trans], dim=-1)

        if self.use_attention:
            if self.norm is not None:
                trans = self.norm(trans)

        trans = einops.rearrange(trans, "n k d -> k n d")

        pe = self.PE_Transformer(src=trans, src_key_padding_mask=mask)

        pe[torch.transpose(mask, 0, 1)] = float('nan')

        # Sum pooling
        pe = torch.nansum(pe, 0, keepdim=False)
        pe = self.proj(pe)

        return pe

        # output = self.re_aggregate_mlp(trans)
        # if self.im_aggregate_mlp is None:
        #     return output

#   JAX implementation:
# # class maglapnet(hk.module):
# #   """for the magnetic laplacian's or combinatorial laplacian's eigenvectors.
# #
# #     args:
# #       d_model_elem: dimension to map each eigenvector.
# #       d_model_aggr: output dimension.
# #       num_heads: number of heads for optional attention.
# #       n_layers: number of layers for mlp/gnn.
# #       dropout_p: dropout for attenion as well as eigenvector embeddings.
# #       activation: element-wise non-linearity.
# #       return_real_output: true for a real number (otherwise complex).
# #       consider_im_part: ignore the imaginary part of the eigenvectors.
# #       use_signnet: if using the sign net idea f(v) + f(-v).
# #       use_gnn: if true use gnn in signnet, otherwise mlp.
# #       use_attention: if true apply attention between eigenvector embeddings for
# #         same node.
# #       concatenate_eigenvalues: if true also concatenate the eigenvalues.
# #       norm: optional norm.
# #       name: name of the layer.
# #   """
# #
# #   def __init__(self,
# #                d_model_elem: int = 32,
# #                d_model_aggr: int = 256,
# #                num_heads: int = 4,
# #                n_layers: int = 1,
# #                dropout_p: float = 0.2,
# #                activation: callable[[tensor], tensor] = jax.nn.relu,
# #                return_real_output: bool = true,
# #                consider_im_part: bool = true,
# #                use_signnet: bool = true,
# #                use_gnn: bool = false,
# #                use_attention: bool = false,
# #                concatenate_eigenvalues: bool = false,
# #                norm: optional[any] = none,
# #                name: optional[str] = none):
# #     super().__init__(name=name)
# #     self.concatenate_eigenvalues = concatenate_eigenvalues
# #     self.consider_im_part = consider_im_part
# #     self.use_signnet = use_signnet
# #     self.use_gnn = use_gnn
# #     self.use_attention = use_attention
# #     self.num_heads = num_heads
# #     self.dropout_p = dropout_p
# #     self.norm = norm
# #
# #     if self.use_gnn:
# #       self.element_gnn = gnn(
# #           int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
# #           gnn_type='gnn',
# #           k_hop=n_layers,
# #           mlp_layers=n_layers,
# #           activation=activation,
# #           use_edge_attr=false,
# #           concat=true,
# #           residual=false,
# #           name='re_element')
# #     else:
# #       self.element_mlp = mlp(
# #           int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
# #           n_layers=n_layers,
# #           activation=activation,
# #           with_norm=false,
# #           final_activation=true,
# #           name='re_element')
# #
# #     self.re_aggregate_mlp = mlp(
# #         d_model_aggr,
# #         n_layers=n_layers,
# #         activation=activation,
# #         with_norm=false,
# #         final_activation=true,
# #         name='re_aggregate')
# #
# #     self.im_aggregate_mlp = none
# #     if not return_real_output and self.consider_im_part:
# #       self.im_aggregate_mlp = mlp(
# #           d_model_aggr,
# #           n_layers=n_layers,
# #           activation=activation,
# #           with_norm=false,
# #           final_activation=true,
# #           name='im_aggregate')
# #
# #   def __call__(self, graph: jraph.graphstuple, eigenvalues: tensor,
# #                eigenvectors: tensor, call_args: callargs) -> tensor:
# #     padding_mask = (eigenvalues > 0)[..., none, :]
# #     padding_mask = padding_mask.at[..., 0].set(true)
# #     attn_padding_mask = padding_mask[..., none] & padding_mask[..., none, :]
# #
# #     trans_eig = jnp.real(eigenvectors)[..., none]
# #     trans_eig = jnp.real(eigenvectors)[..., none]
# #
# #     if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
# #       trans_eig_im = jnp.imag(eigenvectors)[..., none]
# #       trans_eig = jnp.concatenate((trans_eig, trans_eig_im), axis=-1)
# #
# #     if self.use_gnn:
# #       trans = self.element_gnn(
# #           graph._replace(nodes=trans_eig, edges=none), call_args).nodes
# #       if self.use_signnet:
# #         trans_neg = self.element_gnn(
# #             graph._replace(nodes=-trans_eig, edges=none), call_args).nodes
# #         trans += trans_neg
# #     else:
# #       trans = self.element_mlp(trans_eig)
# #       if self.use_signnet:
# #         trans += self.element_mlp(-trans_eig)
# #
# #     if self.concatenate_eigenvalues:
# #       eigenvalues_ = jnp.broadcast_to(eigenvalues[..., none, :],
# #                                       trans.shape[:-1])
# #       trans = jnp.concatenate((eigenvalues_[..., none], trans), axis=-1)
# #
# #     if self.use_attention:
# #       if self.norm is not none:
# #         trans = self.norm()(trans)
# #       attn = multiheadattention(
# #           self.num_heads,
# #           key_size=trans.shape[-1] // self.num_heads,
# #           value_size=trans.shape[-1] // self.num_heads,
# #           model_size=trans.shape[-1],
# #           w_init=none,
# #           dropout_p=self.dropout_p,
# #           with_bias=False)
# #       trans += attn(
# #           trans,
# #           trans,
# #           trans,
# #           mask=attn_padding_mask,
# #           is_training=call_args.is_training)
# #
# #     padding_mask = padding_mask[..., None]
# #     trans = trans * padding_mask
# #     trans = trans.reshape(trans.shape[:-2] + (-1,))
# #
# #     if self.dropout_p and call_args.is_training:
# #       trans = hk.dropout(hk.next_rng_key(), self.dropout_p, trans)
# #
# #     output = self.re_aggregate_mlp(trans)
# #     if self.im_aggregate_mlp is None:
# #       return output
# #
# #     output_im = self.im_aggregate_mlp(trans)
# #     output = output + 1j * output_im
# #     return output
