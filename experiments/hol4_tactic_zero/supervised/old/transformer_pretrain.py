from tqdm import tqdm#%%
from torch.utils.data import DataLoader, TensorDataset
import math

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import json
from data.hol4.ast_def import *
import pickle
from torchtext.vocab import build_vocab_from_iterator


#%%
with open("../../../data/hol4/old/train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)
#%%
#data in polished format with (goal, premise)
#%%
#process data to remove variable/ function variables as in graph


#%%
with open("../../../../data/hol4/data_v2/data/include_probability.json") as f:
    db = json.load(f)
#%%
tokens = list(
        set([token.value for polished_goal in db.keys() for token in polished_to_tokens_2(polished_goal)]))

#%%


# def tokenizer(inp_str): ## This method is one way of creating tokenizer that looks for word tokens
#     return re.findall(r"\w+", inp_str)
#
# tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
#
def build_vocab(l):
    for token in l:
        yield [token]

vocab = build_vocab_from_iterator(build_vocab(tokens), specials=["<UNK>"], min_freq=0)
vocab.set_default_index(vocab["<UNK>"])

with open("../../../data/hol4/data/transformer_vocab.pk", "wb") as f:
    pickle.dump(vocab, f)

train_seq = []

max_len = 256


for i, (goal, premise, y) in enumerate(train):
    train_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))

val_seq = []
for i, (goal, premise, y) in enumerate(val):
    val_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))

test_seq = []
for i, (goal, premise, y) in enumerate(test):
    test_seq.append(([i.value for i in polished_to_tokens_2(goal)], [i.value for i in polished_to_tokens_2(premise)], y))

# goals, premises, targets = list(zip(*train))

train_goals = []
train_premises = []
train_targets = []

for goal, premise, y in train_seq:
    train_goals.append(goal)
    train_premises.append(premise)
    train_targets.append(y)


val_goals = []
val_premises = []
val_targets = []

for goal, premise, y in val_seq:
    val_goals.append(goal)
    val_premises.append(premise)
    val_targets.append(y)

test_goals = []
test_premises = []
test_targets = []

for goal, premise, y in test_seq:
    test_goals.append(goal)
    test_premises.append(premise)
    test_targets.append(y)

def vectorise(goal_list, premise_list, target_list, max_len=256):
    idx_list = [vocab(toks) for toks in goal_list]
    X_G = [sample+([0]* (max_len-len(sample))) if len(sample)<max_len else sample[:max_len] for sample in idx_list]
    idx_list = [vocab(toks) for toks in premise_list]
    X_P = [sample+([0]* (max_len-len(sample))) if len(sample)<max_len else sample[:max_len] for sample in idx_list]
    return torch.tensor(X_G, dtype=torch.int32), torch.tensor(X_P, dtype=torch.int32), torch.tensor(target_list, dtype=torch.long)

train_dataset = vectorise(train_goals, train_premises, train_targets)

val_data = vectorise(val_goals, val_premises, val_targets)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
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

class TransformerEmbedding(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        # self.initial_encoder = inner_embedding_network.F_x_module_(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
def gen_embedding(model, input, src_mask):
    out = model(input, src_mask)
    # print (out.shape)
    out = torch.transpose(out,0,2)
    gmp = nn.MaxPool1d(256, stride=1)
    # ret = torch.cat([gmp(out).squeeze(-1), torch.sum(out,dim=2)], dim = 1)
    #cat global average and max pools as with GNN encoder
    # print (gmp(out).shape)
    return torch.cat([gmp(out).squeeze(-1).transpose(0,1), (torch.sum(out, dim=2)/torch.count_nonzero(out, dim=2)).transpose(0,1)], dim=1)#
        # gmp(out).squeeze(-1).transpose(0,1) #ret


from models.gnn.formula_net import inner_embedding_network


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


#run_edges(1e-3, 0, 20, 1024, 64, 0, False)
#run_2(1e-3, 0, 20, 1024, 64, 4, False)

def accuracy_transformer(model_1, model_2,batch, fc):
    g,p,y = batch
    batch_size = len(g)
    src_mask = generate_square_subsequent_mask(256).to(device)
    g = torch.transpose(g, 0, 1)
    p = torch.transpose(p, 0, 1)
    embedding_1 = gen_embedding(model_1, g.to(device), src_mask)
    embedding_2 = gen_embedding(model_2, p.to(device), src_mask)


    preds = fc(torch.cat([embedding_1, embedding_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(y).to(device)) / len(y)

def run_transformer_pretrain(step_size, decay_rate, num_epochs, batch_size, embedding_dim, n_head=1, n_layers=1, d_hid=64, save=False):

    # loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    # val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    G,P,Y = train_dataset

    dataset = TensorDataset(G,P,Y)
    # batch_size = 50
    loader = DataLoader(dataset, batch_size=batch_size)

    V_G, V_P, V_Y = val_data
    val_dataset = TensorDataset(V_G, V_P, V_Y)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_1 = TransformerEmbedding(ntoken=len(vocab), d_model=128, nhead=n_head, d_hid=d_hid, nlayers=n_layers).to(device)
    model_2 = TransformerEmbedding(ntoken=len(vocab), d_model=128, nhead=n_head, d_hid=d_hid, nlayers=n_layers).to(device)
    fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)

    op_1 =torch.optim.Adam(model_1.parameters(), lr=step_size)
    op_2 =torch.optim.Adam(model_2.parameters(), lr=step_size)
    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    val_losses = []
    best_acc = 0.

    for j in range(num_epochs):
        print (f"Epoch: {j}")

        for batch_idx, (g,p,y) in tqdm(enumerate(loader)):
            # op_enc.zero_grad()
            op_1.zero_grad()
            op_2.zero_grad()
            op_fc.zero_grad()

            src_mask = generate_square_subsequent_mask(256).to(device)

            # if len(g) != batch_size:
            #     src_mask = generate_square_subsequent_mask(len(g)).to(device)

            g = torch.transpose(g, 0,1)
            p = torch.transpose(p, 0,1)
            embedding_1 = gen_embedding(model_1, g.to(device), src_mask)
            embedding_2 = gen_embedding(model_2, p.to(device), src_mask)

            # print (embedding_1.shape, embedding_2.shape)
            preds = fc(torch.cat([embedding_1, embedding_2], axis=1))

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), torch.LongTensor(y).to(device))

            loss.backward()

            op_1.step()
            op_2.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)
            # print (f"training loss: {loss.detach()}")

            if batch_idx % 100 == 0:
                tmp_loss = []
                for _ in range((2048//batch_size)):
                    v_l = iter(val_loader)
                    tmp_loss.append(accuracy_transformer(model_1, model_2, next(v_l), fc).detach())#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations))
                    # validation_loss = accuracy_transformer(model_1, model_2, next(iter(val_loader)), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                # val_losses.append((validation_loss.detach(), j, i))

                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(sum(tmp_loss)/len(tmp_loss)))

                if sum(tmp_loss)/len(tmp_loss) > best_acc:
                    best_acc = sum(tmp_loss)/len(tmp_loss)
                    print (f"New best validation accuracy: {best_acc}")
                    # only save encoder if best accuracy so far
                    if save == True:
                        torch.save(model_1, "../rl/old/model_checkpoints/transformer_encoder_latest_goal_2_2_64_small")
                        torch.save(model_2,
                                   "../rl/old/model_checkpoints/transformer_encoder_latest_premise_2_2_64_small")

    print (f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses

run_transformer_pretrain(1e-3, 0, 200, 1024, 64, n_head=2, n_layers=2, d_hid=64, save=False)

# best run was 1e-3, 0, 40, 32, 64, n_head = 8, n_layers =1, d_hid = 64

#
# if __name__ == "__main__":
#     import sys
#     args = sys.argv[1:]
#     run_transformer_pretrain(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])