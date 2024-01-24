import sys

import einops
from torch import nn

from models.TacticZero import seq2seq
from models.TacticZero.batch_predictor import BatchPredictor
from models.TacticZero.checkpoint import Checkpoint

sys.modules['seq2seq'] = seq2seq

'''

Fixed Autoencoder model from Original TacticZero Paper,
trained to optimise reconstruction loss on sequence text.

'''


class TacticZeroAutoEncoder(nn.Module):
    def __init__(self, checkpoint_path, embedding_dim=256):
        super(TacticZeroAutoEncoder, self).__init__()

        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab

        self.encoder = BatchPredictor(seq2seq, input_vocab, output_vocab)
        self.embedding_dim = embedding_dim

    def forward(self, expr_list):
        return einops.rearrange(self.encoder.encode(expr_list)[0], 'x b d -> b (x d)', x=2, d=self.embedding_dim // 2)
