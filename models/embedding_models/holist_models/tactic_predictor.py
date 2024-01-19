from torch import nn
import einops
import torch


class TacticPredictor(nn.Module):
    def __init__(self, num_tactics, embedding_dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(128, num_tactics))

    def forward(self, x):
        x = self.mlp(x)
        return x


class CombinerNetwork(nn.Module):
    def __init__(self, embedding_dim, num_tactics, tac_embed_dim, dropout=0.3):
        super().__init__()
        self.tac_embedding = nn.Embedding(num_tactics, tac_embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim * 3 + tac_embed_dim, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(128, 1))

    def forward(self, goal, premise, tactic):
        tactic = self.tac_embedding(tactic)
        tactic = einops.repeat(tactic, 'b d -> b k d', k=goal.shape[1])
        x = torch.cat([goal, premise, torch.mul(goal, premise), tactic], dim=-1)
        x = self.mlp(x)
        return x
