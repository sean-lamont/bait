import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextPolicy(nn.Module):
    def __init__(self):
        super(ContextPolicy, self).__init__()
        self.fc = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        self.head = nn.Linear(1024, 1)
        # self.out = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        # x = self.out(self.head(x))
        x = self.head(x)
        return x

class ScorePolicy(nn.Module):
    def __init__(self, out_dim):
        super(ScorePolicy, self).__init__()
        self.fc = nn.Linear(256, 512)
        self.head = nn.Linear(512, out_dim)

        # self.out = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc(x))
        # x = self.out(self.head(x))
        x = self.head(x)
        return x

# input size: (1, 1, 4, 64)
class TacPolicy(nn.Module):
    def __init__(self, tactic_size):
        super(TacPolicy, self).__init__()
        self.fc = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)        
        self.head = nn.Linear(1024, tactic_size)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))        
        x = self.out(self.head(x))
        return x


class ArgPolicy(nn.Module):
    def __init__(self, tactic_size, embedding_dim):
        super(ArgPolicy, self).__init__()
        self.tactic_embeddings = nn.Embedding(tactic_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim)

        self.fc = nn.Linear(embedding_dim+256*2, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.head = nn.Linear(1024, 1)        
        # self.out = nn.Sigmoid()

    # x is the previously predicted argument / tactic.
    # candidates is a matrix of possible arguments concatenated with the hidden states.
    def forward(self, x, candidates, hidden):
        if x.shape == torch.Size([1]):
            x = self.tactic_embeddings(x)#.unsqueeze(0)
        else:
            x = x
        s = F.relu(self.fc(candidates))
        s = F.relu(self.fc2(s))
        # scores = self.out(self.head(s))
        scores = self.head(s)
        o, hidden = self.lstm(x, hidden)
        
        return hidden, scores


class TermPolicy(nn.Module):
    def __init__(self, tactic_size, embedding_dim):
        super(TermPolicy, self).__init__()
        self.tactic_embeddings = nn.Embedding(tactic_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim+256*2, 512)
        self.fc2 = nn.Linear(512, 1024)        
        self.head = nn.Linear(1024, 1)
        # self.out = nn.Sigmoid()

    def forward(self, candidates, tac):
        tac = self.tactic_embeddings(tac).view(1,-1)
        tac_tensor = torch.cat([tac for _ in range(candidates.shape[0])])
        x = torch.cat([candidates, tac_tensor], dim=1)
        
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))        
        # x = self.out(self.head(x))
        x = self.head(x)
        return x
