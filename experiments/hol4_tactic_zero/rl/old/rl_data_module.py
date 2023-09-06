from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as loader
import lightning.pytorch as pl
import torch.optim
from data.hol4.ast_def import graph_to_torch_labelled
from data.hol4 import ast_def
from environments.hol4.new_env import *
from data.utils.graph_data_utils import ptr_to_complete_edge_index


def data_to_relation(batch):
    xis = []
    xjs = []
    edge_attrs = []
    for graph in batch:
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        xi = torch.index_select(x, 0, edge_index[0])
        xj = torch.index_select(x, 0, edge_index[1])
        xis.append(xi)
        xjs.append(xj)
        edge_attrs.append(edge_attr.long())

    xi = torch.nn.utils.rnn.pad_sequence(xis)
    xj = torch.nn.utils.rnn.pad_sequence(xjs)
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attrs)

    mask= (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    return Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)

def to_sequence(data, vocab):
    data = data.split(" ")
    data = torch.LongTensor([vocab[c] for c in data if c in vocab])
    return data


def collate_pad(batch):
    x = torch.nn.utils.rnn.pad_sequence(batch)
    x = x[:300]
    mask = (x == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)
    return Data(data=x, mask=mask)

def gather_encoded_content_gnn(history, encoder, device, graph_db, token_enc, data_type='graph'):
    fringe_sizes = []
    contexts = []
    reverted = []
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        g = revert_with_polish(e)
        reverted.append(g)

    # def hack(x):
    #     if data_type == 'graph' or data_type == 'relation' or data_type == 'sat':
    #         if x in graph_db:
    #             return graph_db[x]
    #         if x not in tmp:
    #             tmp[x] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(x), token_enc)
    #         return tmp[x]
    #     else:
    #         db, vocab = graph_db
    #         if x in db:
    #             return db[x]
    #         if x not in tmp:
    #             tmp[x] = to_sequence(x,vocab)
    #         return tmp[x]
    #

    if data_type == 'graph':
        # graphs = [hack(t) for t in reverted]

        graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc) for t in reverted]
        loader = DataLoader(graphs, batch_size=len(graphs))
        batch = next(iter(loader))
        batch.to(device)
        batch.edge_attr = batch.edge_attr.long()

        # for batch norm, when only one goal
        # encoder.eval()
        representations = torch.unsqueeze(encoder(batch), 1)
        # encoder.train()

    elif data_type == 'sat':
        # todo listcomp takes ages below with graph_to_torch.. maybe add to graph_db every new expression, or to a tmp_db for a given goal?
        # graphs = [hack(t) for t in reverted]

        graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc) for t in reverted]
        loader = DataLoader(graphs, batch_size=len(graphs))
        batch = next(iter(loader))
        batch.edge_attr = batch.edge_attr.long()
        batch.attention_edge_index = ptr_to_complete_edge_index(batch.ptr)
        batch.to(device)
        representations = torch.unsqueeze(encoder(batch), 1)


    elif data_type == 'relation':
        graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc) for t in reverted]
        graphs = data_to_relation(graphs)
        graphs.to(device)
        representations = torch.unsqueeze(encoder(graphs), 1)

    elif data_type == 'sequence':
        db, vocab = graph_db
        batch = [db[t] if t in db.keys() else to_sequence(t, vocab) for t in reverted]
        batch = [to_sequence(t, vocab) for t in reverted]
        # batch = [hack(t) for t in reverted]
        data = collate_pad(batch).to(device)
        representations = torch.unsqueeze(encoder(data), 1)


    return representations, contexts, fringe_sizes

class RLData(pl.LightningDataModule):
    def __init__(self, train_goals, test_goals, config, database=None, graph_db=None):
        super().__init__()
        self.config = config
        self.env = HolEnv("T")
        self.train_goals = train_goals
        self.test_goals = test_goals
        self.database = database
        self.graph_db = graph_db

    def gen_fact_pool(self, env, goal):
        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', goal[0])))
        goal_theory = self.database[goal[0]][0]
        polished_goal = env.fringe["content"][0]["polished"]["goal"]
        try:
            allowed_arguments_ids = []
            candidate_args = []
            for i, t in enumerate(self.database):
                theory_allowed = self.database[t][0] in allowed_theories
                diff_theory = self.database[t][0] != goal_theory
                prev_theory = int(self.database[t][3]) < int(self.database[polished_goal][3])
                if theory_allowed and (diff_theory or prev_theory):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)
            env.toggle_simpset("diminish", goal_theory)
            # print("Removed simpset of {}".format(goal_theory))

        except Exception as e:
            raise Exception(f"Error generating fact pool: {e}")

        if self.config['data_type'] == 'graph':
            allowed_fact_batch = []
            graphs = [self.graph_db[t] for t in candidate_args]
            # loader = DataLoader(graphs, batch_size=len(candidate_args))
            # loader = DataLoader(graphs, batch_size=32,drop_last=False)

            loader = DataLoader(graphs, batch_size=len(candidate_args))
            allowed_fact_batch = next(iter(loader))
            allowed_fact_batch.edge_attr = allowed_fact_batch.edge_attr.long()

            # for batch in loader:
            #     batch.edge_attr = batch.edge_attr.long()
            #     allowed_fact_batch.append(batch)

        elif self.config['data_type'] == 'relation':
            graphs = [self.graph_db[t] for t in candidate_args]
            graphs = [graphs[i:i + 32] for i in range(0, len(graphs), 32)]
            allowed_fact_batch = [data_to_relation(graph) for graph in graphs]

        elif self.config['data_type'] == 'sequence':
            db, vocab = self.graph_db
            batches = [to_sequence(t, vocab) for t in candidate_args]
            batches = [batches[i:i + 32] for i in range(0, len(batches), 32)]
            allowed_fact_batch = [collate_pad(batch) for batch in batches]


        elif self.config['data_type'] == 'sat':
            allowed_fact_batch = []
            graphs = [self.graph_db[t] for t in candidate_args]
            loader = DataLoader(graphs, batch_size=32,drop_last=False)

            for batch in loader:
                batch.edge_attr = batch.edge_attr.long()
                batch.attention_edge_index = ptr_to_complete_edge_index(batch.ptr)
                allowed_fact_batch.append(batch)


        #todo not sure if we need allowed_arguments_ids?
        return allowed_fact_batch, allowed_arguments_ids, candidate_args

    def setup_goal(self, goal):
        goal = goal[0]
        try:
            self.env.reset(goal[1])
        except:
            self.env = HolEnv("T")
            return None
        allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(self.env, goal)
        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, self.env

    def train_dataloader(self):
        return loader(self.train_goals, batch_size=1, collate_fn=self.setup_goal)

    def val_dataloader(self):
        return loader(self.test_goals, batch_size=1, collate_fn=self.setup_goal)

    # todo: val set to terminate training with??
    # def test_dataloader(self):
    #     return loader(self.test_goals, collate_fn=self.setup_goal)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        try:
            goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
            # allowed_fact_batch = [x.to(device) for x in allowed_fact_batch]
        except Exception as e:
            print (e)
            return None

        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env



# test model
# module = RLData(train_goals=train_goals, test_goals=test_goals, database=compat_db, graph_db=graph_db)
# module.setup("fit")
# batch = next(iter(module.train_dataloader()))
# print (batch)
