import sys

import einops
from torch import nn

from models.tactic_zero import seq2seq
from models.tactic_zero.batch_predictor import BatchPredictor
from models.tactic_zero.checkpoint import Checkpoint

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

        # out, sizes = self.encoder.encode(expr_list)
        # merge two hidden variables
        # representations = torch.cat(out.split(1), dim=2).squeeze(0)
        #
        # reps = []
        #
        # for expr in expr_list:
        #     expr = expr.strip().split()
        #     out, _ = self.encoder.encode([expr])
        #     out = einops.rearrange(out, 'b 1 d -> 1 (b d)')
        #     reps.append(out)
        #
        # return torch.cat(reps, dim=0)

        # return torch.cat(
        #     [einops.rearrange(self.encoder.encode()[0], 'b 1 d -> 1 (b d)') for expr in
        #      expr_list], dim=0)
