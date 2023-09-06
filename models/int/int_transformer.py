import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import einops


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        # add 1 in case of CLS token
        max_len = max_len + 1
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


"""

Wrapper for transformer taking in tuple with first element as data, second as mask

"""


class INTTransformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, enc=True, in_embed=False,
                 max_len=512):

        super().__init__()

        self.transformer_embedding = TransformerEmbedding(ntoken=None, d_model=d_model, nhead=nhead, d_hid=d_hid,
                                                          nlayers=nlayers, dropout=dropout, enc=enc,
                                                          global_pool=False, in_embed=in_embed, max_len=max_len)

        self.embedding = nn.Embedding(ntoken, d_model, padding_idx=0)

    def forward(self, data, iv_data, mask, global_pool=False):
        # if type(data) == list or type(data) == tuple:
        #     x = data[0]
        #     mask = data[1]
        # else:
        #     x = data.data
        #     mask = data.mask

        iv_data = self.embedding(iv_data)
        x = self.embedding(data)
        x = x + iv_data

        out = self.transformer_embedding(x, mask)

        if global_pool:
            return torch.sum(out, dim=1)[0]

        return out





"""

Transformer Embedding 

"""


class TransformerEmbedding(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, enc=True, in_embed=True, global_pool=True, max_len=512):
        super().__init__()
        self.in_embed = in_embed
        self.enc = enc
        self.global_pool = global_pool
        self.model_type = 'Transformer'

        if self.enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=max_len)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if self.in_embed:
            self.encoder = nn.Embedding(ntoken, d_model)
            self.init_weights()

        self.d_model = d_model

        # self.cls_token = nn.Parameter(torch.randn(1, d_model))

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data, mask=None):
        src = data

        if self.in_embed:
            src = self.encoder(src) * math.sqrt(self.d_model)

        if self.enc:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_key_padding_mask=mask)  # , memory_mask=mask)

        output = torch.transpose(output, 0, 1)

        if self.global_pool:
            # CLS token value
            output = output[:, 0]
            return output

        return output
