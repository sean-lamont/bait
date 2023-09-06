import os
import torch
import torch_geometric.loader.dataloader
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from tqdm import tqdm



def get_directed_edge_index(num_nodes, edge_idx):
    if num_nodes == 1:
        return torch.LongTensor([[], []])

    from_idx = []
    to_idx = []

    for i in range(0, num_nodes):
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                                  edge_index=edge_idx)
        except:
            print(f"exception {i, num_nodes, edge_idx}")

        found_nodes = list(ancestor_nodes.numpy())
        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,
                                                                              edge_index=edge_idx,
                                                                              flow='target_to_source')

        found_nodes = list(children_nodes.numpy())
        if i in found_nodes:
            found_nodes.remove(i)

        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


# probably slow, could recursively do k-hop subgraph with k = 1 instead
def get_depth_from_graph(num_nodes, edge_index):
    to_idx = edge_index[1]

    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i,
                                                                          edge_index=edge_index,
                                                                          flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes

    return depths


class DirectedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'attention_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


'''
Relation Attention Data Module
'''


class HOL4DataModule(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            expr = db.expression_graph_data
            meta = db.expression_metadata

            graph_dict = {v['_id']: v['graph'] for v in expr.find({})}
            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression']) for v in
                         meta.find({})}

            train_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in
                          split.find({}) if
                          v['split'] == 'train']
            val_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in split.find({})
                        if
                        v['split'] == 'val']
            test_data = [self.to_relation((graph_dict[v['conj']], graph_dict[v['stmt']], v['y'])) for v in
                         split.find({}) if
                         v['split'] == 'test']

            data = {'graph_dict': graph_dict, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}
            torch.save(data, self.dir + "/data.pt")

    def to_relation(self, data):
        data_1, data_2, y = data

        x = data_1['onehot']
        edge_index = data_1['edge_index']
        edge_attr = torch.LongTensor(data_1['edge_attr']) + 1
        xi = torch.LongTensor([x[i] for i in edge_index[0]]) + 1
        xj = torch.LongTensor([x[i] for i in edge_index[1]]) + 1
        data_1 = Data(xi=xi, xj=xj, edge_attr_=edge_attr)

        x = data_2['onehot']
        edge_index = data_2['edge_index']
        edge_attr = torch.LongTensor(data_2['edge_attr']) + 1
        xi = torch.LongTensor([x[i] for i in edge_index[0]]) + 1
        xj = torch.LongTensor([x[i] for i in edge_index[1]]) + 1
        data_2 = Data(xi=xi, xj=xj, edge_attr_=edge_attr)

        return data_1, data_2, y

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def collate_pad(self, batch):
        xis1 = []
        xis2 = []
        xjs1 = []
        xjs2 = []
        edge_attrs1 = []
        edge_attrs2 = []
        ys = []

        for data_1, data_2, y in batch:
            xis1.append(data_1.xi)
            xjs1.append(data_1.xj)
            edge_attrs1.append(data_1.edge_attr_)
            xis2.append(data_2.xi)
            xjs2.append(data_2.xj)
            edge_attrs2.append(data_2.edge_attr_)
            ys.append(y)

        xi1 = torch.nn.utils.rnn.pad_sequence(xis1)
        xj1 = torch.nn.utils.rnn.pad_sequence(xjs1)
        edge_attr_1 = torch.nn.utils.rnn.pad_sequence(edge_attrs1)

        mask1 = (xi1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        xi2 = torch.nn.utils.rnn.pad_sequence(xis2)
        xj2 = torch.nn.utils.rnn.pad_sequence(xjs2)
        edge_attr_2 = torch.nn.utils.rnn.pad_sequence(edge_attrs2)

        mask2 = (xi2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return Data(xi=xi1, xj=xj1, edge_attr_=edge_attr_1, mask=mask1), Data(xi=xi2, xj=xj2, edge_attr_=edge_attr_2,
                                                                              mask=mask2), torch.LongTensor(ys)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_pad)


'''

Data Module for Graph based Models (Currently GNNs and Structure Aware Attention)

'''


class HOL4DataModuleGraph(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            expr = db.expression_graph_data
            meta = db.expression_metadata

            graph_dict = {v['_id']: v['graph'] for v in expr.find({})}
            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression']) for v in
                         meta.find({})}

            train_data = [(v['conj'], v['stmt'], v['y']) for v in tqdm(split.find({}))
                          if v['split'] == 'train']

            val_data = [(v['conj'], v['stmt'], v['y']) for v in tqdm(split.find({}))
                        if v['split'] == 'valid']

            test_data = [(v['conj'], v['stmt'], v['y']) for v in tqdm(split.find({}))
                         if v['split'] == 'test']

            # add attention edge index
            print("Adding attention edge index")
            for k, v in tqdm(graph_dict.items()):
                v['attention_edge_index'] = get_directed_edge_index(len(v['onehot']), torch.LongTensor(v['edge_index']))
                v['depth'] = get_depth_from_graph(len(v['onehot']), torch.LongTensor(v['edge_index']))
                graph_dict[k] = v

            data = {'graph_dict': graph_dict, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}

            torch.save(data, self.dir + "/data.pt")


    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        self.graph_dict = self.data['graph_dict']
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def attention_collate(self, batch):
        y = torch.LongTensor([b[2] for b in batch])
        data_1 = [b[0] for b in batch]
        data_2 = [b[1] for b in batch]

        data_1 = [DirectedData(x=torch.LongTensor(self.graph_dict[d]['onehot']),
                                                    edge_index=torch.LongTensor(self.graph_dict[d]['edge_index']),
                                                    edge_attr=torch.LongTensor(self.graph_dict[d]['edge_attr']), )
                                       # attention_edge_index=torch.LongTensor(
                                       #     self.graph_dict[d]['attention_edge_index']),
                                       # abs_pe=torch.LongTensor(self.graph_dict[d]['depth']))
                                       for d in data_1]

        data_1 = Batch.from_data_list(data_1)

        data_2 = [DirectedData(x=torch.LongTensor(self.graph_dict[d]['onehot']),
                                                    edge_index=torch.LongTensor(self.graph_dict[d]['edge_index']),
                                                    edge_attr=torch.LongTensor(self.graph_dict[d]['edge_attr']), )
                                       # attention_edge_index=torch.LongTensor(
                                       #     self.graph_dict[d]['attention_edge_index']),
                                       # abs_pe=torch.LongTensor(self.graph_dict[d]['depth']))
                                       for d in data_2]

        data_2 = Batch.from_data_list(data_2)

        data_1.attention_edge_index = ptr_to_complete_edge_index(data_1.ptr)
        data_2.attention_edge_index = ptr_to_complete_edge_index(data_2.ptr)

        return data_1, data_2, y

    def train_dataloader(self):
        # return torch_geometric.loader.dataloader.DataLoader(self.train_data, batch_size=self.batch_size)

        return torch.utils.data.dataloader.DataLoader(self.train_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate)

    def val_dataloader(self):
        # return torch_geometric.loader.dataloader.DataLoader(self.val_data, batch_size=self.batch_size)
        return torch.utils.data.dataloader.DataLoader(self.val_data, batch_size=self.batch_size,
                                                      collate_fn=self.attention_collate)

    def test_dataloader(self):
        return torch_geometric.loader.dataloader.DataLoader(self.test_data, batch_size=self.batch_size)

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        data_1, data_2, y = batch

        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        y = y.to(device)

        return data_1, data_2, y


'''

 Sequence data class with all tokens (including e.g @ symbols) and keeping position
 
'''


class HOL4SequenceModule(LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol4_tactic_zero"
            db = MongoClient()
            db = db[db_name]

            split = db.pretrain_data
            meta = db.expression_metadata

            expr_dict = {v['_id']: (v['theory'], v['name'], v['dep_id'], v['type'], v['plain_expression'])
                         for v in meta.find({})}

            # build vocab
            vocab = {}
            i = 1
            for k in expr_dict.keys():
                tmp = k.split(" ")
                for t in tmp:
                    if t not in vocab:
                        vocab[t] = i
                        i += 1

            train_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                          v['split'] == 'train']
            val_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                        v['split'] == 'val']
            test_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({}) if
                         v['split'] == 'test']

            data = {'vocab': vocab, 'expr_dict': expr_dict, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}
            torch.save(data, self.dir + "/data.pt")

    def to_sequence(self, data, vocab):
        data_1, data_2, y = data

        data_1 = data_1.split(" ")
        data_1 = torch.LongTensor([vocab[c] for c in data_1])

        data_2 = data_2.split(" ")
        data_2 = torch.LongTensor([vocab[c] for c in data_2])

        return data_1, data_2, y

    def setup(self, stage: str) -> None:
        print("Setting up data loaders..")
        self.data = torch.load(self.dir + "/data.pt")
        if stage == "fit":
            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
        if stage == "test":
            self.test_data = self.data['test_data']

    def collate_pad(self, batch):
        x1 = []
        x2 = []
        ys = []
        for data_1, data_2, y in batch:
            x1.append(data_1)
            x2.append(data_2)
            ys.append(y)

        x1 = torch.nn.utils.rnn.pad_sequence(x1)
        x1 = x1[:300]
        mask1 = (x1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        x2 = torch.nn.utils.rnn.pad_sequence(x2)
        x2 = x2[:300]
        mask2 = (x2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return (x1, mask1), (x2, mask2), torch.LongTensor(ys)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_pad)
