import torchdata
import time
from torch.utils.data import DataLoader
import re
import os
from torch_geometric.data import Data, Batch
from data.utils.hd5_utils import to_hdf5
import h5py
import lightning.pytorch as pl
from utils.mongodb_utils import get_batches
from tqdm import tqdm
from pymongo import MongoClient
import torch

import warnings

warnings.filterwarnings('ignore')


class DirectedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'attention_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class LinkData(Data):
    def __init__(self, edge_index_s=None,
                 x_s=None, edge_index_t=None,
                 x_t=None, edge_attr_s=None,
                 edge_attr_t=None,
                 softmax_idx_s=None,
                 softmax_idx_t=None,
                 y=None):

        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t
        # softmax index used for AMR model
        self.softmax_idx_s = softmax_idx_s
        self.softmax_idx_t = softmax_idx_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s' or key == 'attention_edge_index_s':
            return self.x_s.size(0)
        elif key == 'edge_index_t' or key == 'attention_edge_index_t':
            return self.x_t.size(0)
        elif key == 'softmax_idx_s':
            return self.softmax_idx_s
        elif key == 'softmax_idx_t':
            return self.softmax_idx_t
        return super().__inc__(key, value, *args, **kwargs)


'''
Load Data from a Mongo cursor. Uses an intermediate buffer to minimise queries
'''


class MongoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cursor, buf_size):
        super(MongoDataset).__init__()
        self.cursor = cursor
        self.batches = get_batches(self.cursor, batch_size=buf_size)
        self.curr_batches = next(self.batches)
        self.remaining = len(self.curr_batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            self.curr_batches = next(self.batches)
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            return self.curr_batches.pop()
        else:
            raise StopIteration


class MongoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        client = MongoClient()
        self.db = client[self.config['db_name']]
        self.collection = self.db[self.config['collection_name']]
        self.batch_size = self.config['batch_size']
        self.options = config['options']

    def sample_to_link(self, sample):
        options = self.options
        stmt_graph = sample['stmt_graph']
        conj_graph = sample['conj_graph']
        y = sample['y']

        x1 = conj_graph['onehot']
        x1_mat = torch.LongTensor(x1)

        x2 = stmt_graph['onehot']
        x2_mat = torch.LongTensor(x2)

        ret = LinkData(x_s=x2_mat, x_t=x1_mat, y=torch.tensor(y))

        if 'edge_index' in options:
            if 'edge_index' in conj_graph and 'edge_index' in stmt_graph:
                x1_edge_index = conj_graph['edge_index']
                x1_edge_index = torch.LongTensor(x1_edge_index)

                x2_edge_index = stmt_graph['edge_index']
                x2_edge_index = torch.LongTensor(x2_edge_index)

                ret.edge_index_t = x1_edge_index
                ret.edge_index_s = x2_edge_index
            else:
                raise NotImplementedError

        if 'edge_attr' in options:
            if 'edge_attr' in conj_graph and 'edge_attr' in stmt_graph:
                x1_edge_attr = conj_graph['edge_attr']
                x1_edge_attr = torch.LongTensor(x1_edge_attr)

                x2_edge_attr = stmt_graph['edge_attr']
                x2_edge_attr = torch.LongTensor(x2_edge_attr)

                ret.edge_attr_t = x1_edge_attr
                ret.edge_attr_s = x2_edge_attr
            else:
                raise NotImplementedError

        # Edge index used to determine where attention is propagated in Message Passing Attention schemes

        if 'attention_edge_index' in options:
            if 'attention_edge_index' in conj_graph and 'attention_edge_index' in stmt_graph:
                ret.attention_edge_index_t = conj_graph['attention_edge_index']
                ret.attention_edge_index_s = stmt_graph['attention_edge_index']
            else:
                ret.attention_edge_index_t = torch.cartesian_prod(torch.arange(x1_mat.size(0)),
                                                                  torch.arange(x1_mat.size(0))).transpose(0, 1)

                ret.attention_edge_index_s = torch.cartesian_prod(torch.arange(x2_mat.size(0)),
                                                                  torch.arange(x2_mat.size(0))).transpose(0, 1)

        if 'softmax_idx' in options:
            ret.softmax_idx_t = x1_edge_index.size(1)
            ret.softmax_idx_s = x2_edge_index.size(1)

        return ret

    def custom_collate(self, data):
        data_list = [self.sample_to_link(d) for d in data]
        batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_t'])
        return separate_link_batch(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # batch = batch[0]
        data_1, data_2, y = batch

        data_1.x = data_1.x.to(device)
        data_2.x = data_2.x.to(device)

        data_1.edge_index = data_1.edge_index.to(device)
        data_2.edge_index = data_2.edge_index.to(device)

        data_1.edge_attr = data_1.edge_attr.to(device)
        data_2.edge_attr = data_2.edge_attr.to(device)

        y = y.to(device)
        # batch.edge_index = batch.edge_index.to(device)
        return data_1, data_2, y

    def setup(self, stage: str):
        if stage == "fit":
            self.train_cursor = self.collection.find({"split": "train"}).sort("rand_idx", 1)
            self.train_data = MongoDataset(self.train_cursor, self.config['buf_size'])

            self.val_cursor = self.collection.find({"split": "valid"}).sort("rand_idx", 1)
            self.val_data = MongoDataset(self.val_cursor, self.config['buf_size'])

        if stage == "test":
            self.test_cursor = self.collection.find({"split": "test"}).sort("rand_idx", 1)
            self.test_data = MongoDataset(self.test_cursor, self.config['buf_size'])

        # if stage == "predict":

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate,
                                           num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)


def separate_link_batch(batch):
    # assume data will always have at least x variable
    data_1 = Data(x=batch.x_t, batch=batch.x_t_batch, ptr=batch.x_t_ptr)
    data_2 = Data(x=batch.x_s, batch=batch.x_s_batch, ptr=batch.x_s_ptr)

    if hasattr(batch, 'edge_index_t'):
        data_1.edge_index = batch.edge_index_t
        data_2.edge_index = batch.edge_index_s

    if hasattr(batch, 'softmax_idx_t'):
        data_1.softmax_idx = batch.softmax_idx_t
        data_2.softmax_idx = batch.softmax_idx_s

    if hasattr(batch, 'edge_attr_t'):
        data_1.edge_attr = batch.edge_attr_t.long()
        data_2.edge_attr = batch.edge_attr_s.long()

    if hasattr(batch, 'attention_edge_index_t'):
        data_1.attention_edge_index = batch.attention_edge_index_t
        data_2.attention_edge_index = batch.attention_edge_index_s

    return data_1, data_2, batch.y


# data_dir = '/home/sean/Documents/phd/repo/aitp/data/utils/holstep_full'

# save preprocessed graph data in batches to files. Prevents need for recomputing batch level attributes,
# such as ptr, batch idx etc.
def write_mongo_to_h5(data_dir):
    data_module = MongoDataModule(config={'buf_size': 2048, 'batch_size': 32, 'db_name': 'hol_step',
                                          'collection_name': 'pretrain_graphs',
                                          'options': ['edge_attr', 'edge_index', 'softmax_idx',
                                                      'attention_edge_index']})

    data_module.setup('fit')
    data_module.setup('test')

    train_loader = iter(data_module.train_dataloader())
    val_loader = iter(data_module.val_dataloader())
    test_loader = iter(data_module.test_dataloader())

    BATCHES_PER_FILE = 512

    def batch_to_h5(loader, name):
        data_list = []
        file_num = 0

        for i, batch in tqdm(enumerate(loader)):
            data_list.append(batch)

            if i > 0 and i % BATCHES_PER_FILE == 0:
                with h5py.File(data_dir + f'/{name}_{file_num}.h5', 'w') as f:
                    to_hdf5(data_list, f, 'data')
                data_list = []
                file_num += 1

    batch_to_h5(train_loader, 'train')
    batch_to_h5(val_loader, 'val')
    batch_to_h5(test_loader, 'test')


@torchdata.datapipes.functional_datapipe("load_h5_data")
class H5DataLoader(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, source_datapipe, cycle=False, **kwargs) -> None:
        if cycle:
            self.source_datapipe = source_datapipe.cycle()
        else:
            self.source_datapipe = source_datapipe
        # self.transform = kwargs['transform']

    # def file_to_data(self, h5file):

    def __iter__(self):
        for file_name in self.source_datapipe:
            with h5py.File(file_name, 'r') as h5file:
                x1 = torch.from_numpy(h5file['data/x1'][:])
                edge_index1 = torch.from_numpy(h5file['data/edge_index1'][:])  # .long()
                edge_attr1 = torch.from_numpy(h5file['data/edge_attr1'][:])  # .long()
                batch1 = torch.from_numpy(h5file['data/batch1'][:])  # .long()

                # ptr1 = torch.from_numpy(h5file['data/ptr1'][:]).tolist()
                ptr1 = torch.from_numpy(h5file['data/ptr1'][:])  # .long()

                edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:]).tolist()
                # edge_ptr1 = torch.from_numpy(h5file['data/edge_ptr1'][:]).long()
                # attention_edge_1 = torch.from_numpy(h5file['data/attention_edge1'][:])

                x2 = torch.from_numpy(h5file['data/x2'][:])  # .long()
                edge_index2 = torch.from_numpy(h5file['data/edge_index2'][:])  # .long()
                edge_attr2 = torch.from_numpy(h5file['data/edge_attr2'][:])  # .long()
                batch2 = torch.from_numpy(h5file['data/batch2'][:])  # .long()

                # ptr2 = torch.from_numpy(h5file['data/ptr2'][:]).tolist()
                ptr2 = torch.from_numpy(h5file['data/ptr2'][:])  # .long()

                edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:]).tolist()
                # edge_ptr2 = torch.from_numpy(h5file['data/edge_ptr2'][:]).long()

                # attention_edge_2 = torch.from_numpy(h5file['data/attention_edge2'][:])

                y = torch.from_numpy(h5file['data/y'][:])  # .long()

                data_len_1 = torch.from_numpy(h5file['data/x1_len'][:])
                data_len_2 = torch.from_numpy(h5file['data/x2_len'][:])

                for i in range(len(x1)):
                    num_nodes1 = data_len_1[i][0]
                    num_edges1 = data_len_1[i][1]

                    num_nodes2 = data_len_2[i][0]
                    num_edges2 = data_len_2[i][1]

                    data_1 = Data(x=x1[i, :num_nodes1],
                                  edge_index=edge_index1[i, :, :num_edges1],
                                  edge_attr=edge_attr1[i, :num_edges1],
                                  batch=batch1[i, :num_nodes1],
                                  ptr=ptr1[i],
                                  # attention_edge_index=attention_edge_1[i],
                                  # attention_edge_index=attention_edge_1[i],
                                  attention_edge_index=ptr_to_complete_edge_index(ptr1[i]),
                                  softmax_idx=edge_ptr1[i])

                    data_2 = Data(x=x2[i, :num_nodes2],
                                  edge_index=edge_index2[i, :, :num_edges2],
                                  edge_attr=edge_attr2[i, :num_edges2],
                                  batch=batch2[i, :num_nodes2],
                                  ptr=ptr2[i],
                                  # attention_edge_index=attention_edge_2[i],
                                  attention_edge_index=ptr_to_complete_edge_index(ptr2[i]),
                                  softmax_idx=edge_ptr2[i])

                    y_i = y[i]

                    yield data_1, data_2, y_i


def build_h5_datapipe(masks, data_dir, cycle=False):
    datapipe = torchdata.datapipes.iter.FileLister(data_dir, masks=masks)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.load_h5_data(cycle=cycle)
    return datapipe


# todo move relation to disk with h5, very cpu intensive
def collate_to_relation(batch):
    data_1, data_2, y = batch[0]

    x = data_1.x + 1
    edge_index = data_1.edge_index

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    xi = torch.tensor_split(xi, data_1.softmax_idx[1:-1])
    xj = torch.tensor_split(xj, data_1.softmax_idx[1:-1])

    edge_attr_ = data_1.edge_attr.long()
    edge_attr_ = torch.tensor_split(edge_attr_, data_1.softmax_idx[1:-1])
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attr_)

    xi = torch.nn.utils.rnn.pad_sequence(xi)
    xj = torch.nn.utils.rnn.pad_sequence(xj)

    xi = xi[:300]
    xj = xj[:300]
    edge_attr_ = edge_attr_[:300]

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    data_1 = Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)

    x = data_2.x + 1
    edge_index = data_2.edge_index

    xi = torch.index_select(x, 0, edge_index[0])
    xj = torch.index_select(x, 0, edge_index[1])

    xi = torch.tensor_split(xi, data_2.softmax_idx[1:-1])
    xj = torch.tensor_split(xj, data_2.softmax_idx[1:-1])

    edge_attr_ = data_2.edge_attr.long()
    edge_attr_ = torch.tensor_split(edge_attr_, data_2.softmax_idx[1:-1])
    edge_attr_ = torch.nn.utils.rnn.pad_sequence(edge_attr_)

    xi = torch.nn.utils.rnn.pad_sequence(xi)
    xj = torch.nn.utils.rnn.pad_sequence(xj)

    xi = xi[:300]
    xj = xj[:300]
    edge_attr_ = edge_attr_[:300]

    mask = (xi == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

    data_2 = Data(xi=xi, xj=xj, edge_attr_=edge_attr_, mask=mask)

    return data_1, data_2, y


class H5DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['data_dir']

    def prepare_data(self):
        print("Preparing dataset..")
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
            write_mongo_to_h5(self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_pipe = build_h5_datapipe("train*", self.data_dir)
            # self.val_pipe = build_h5_datapipe("val*",self.data_dir, cycle=True)
            self.val_pipe = build_h5_datapipe("val*", self.data_dir, cycle=True)
        if stage == "test":
            self.test_pipe = build_h5_datapipe("test*", self.data_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_pipe,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            # pin_memory=True,
            num_workers=0, collate_fn=lambda x: x)
        # num_workers = 0, collate_fn = collate_to_relation)

    def val_dataloader(self):
        # cycle through
        return torch.utils.data.DataLoader(
            dataset=self.val_pipe,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0, collate_fn=lambda x: x)
        # num_workers = 0, collate_fn = collate_to_relation)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_pipe,
            batch_size=1,
            # shuffle=True,
            drop_last=True,
            num_workers=0, collate_fn=lambda x: x)
        # num_workers = 0, collate_fn = collate_to_relation)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # batch[0] when relation
        data_1, data_2, y = batch[0]
        # data_1, data_2, y = batch
        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        y = y.to(device)
        return data_1, data_2, y


def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


_TOKEN_RE = re.compile(r'\\//|/\\|\\|\?\!|[@.?!(),]|[^\s?!(),.]+')


class HOLStepSequenceModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            print("Generating data from MongoDB..")
            os.mkdir(self.dir)
            db_name = "hol_step"
            db = MongoClient()
            db = db[db_name]

            split = db.train_val_test_data
            expr = db.expression_graphs

            expressions = [v['_id'] for v in expr.find({})]

            # build vocab
            count = {}
            i = 1
            for k in expressions:
                tmp = _TOKEN_RE.findall(k)
                for t in tmp:
                    if t not in count:
                        count[t] = 1
                    else:
                        count[t] = count[t] + 1

            # take most frequent 2000 characters, set 0 as pad index, 1 as UNK
            counts = list(count.items())
            counts = sorted(counts, key=lambda x: x[1], reverse=True)

            top_k = [x[0] for x in counts[:2000]]

            print(
                f"Top 2000 keys covers {sum([x[1] for x in counts[:2000]]) * 100 / sum([x[1] for x in counts])}% of tokens")

            vocab = {}
            i = 2
            for tok in top_k:
                vocab[tok] = i
                i += 1

            for tok in counts[2000:]:
                vocab[tok[0]] = 1

            train_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in
                          tqdm(split.find({'split': 'train'}))]
            val_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({'split': 'valid'})]
            test_data = [self.to_sequence((v['conj'], v['stmt'], v['y']), vocab) for v in split.find({'split': 'test'})]

            data = {'vocab': vocab, 'train_data': train_data, 'val_data': val_data,
                    'test_data': test_data}

            print(f'Saving data to {self.dir}/data.pt')
            torch.save(data, self.dir + "/data.pt")

    def to_sequence(self, data, vocab):
        data_1, data_2, y = data

        data_1 = _TOKEN_RE.findall(data_1)
        data_1 = torch.LongTensor([vocab[c] for c in data_1])

        data_2 = _TOKEN_RE.findall(data_2)
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
        x1 = x1[:2048]
        mask1 = (x1 == 0).T
        mask1 = torch.cat([mask1, torch.zeros(mask1.shape[0]).bool().unsqueeze(1)], dim=1)

        x2 = torch.nn.utils.rnn.pad_sequence(x2)
        x2 = x2[:2048]
        mask2 = (x2 == 0).T
        mask2 = torch.cat([mask2, torch.zeros(mask2.shape[0]).bool().unsqueeze(1)], dim=1)

        return (x1, mask1), (x2, mask2), torch.LongTensor(ys)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_pad)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_pad)


'''

Data Module for HOLStep with attention edges for masking Attention. As the data is too large to fit in memory, 
MongoDB is used to find examples and then retrieve the associated graph data from there

'''

class MongoDataModuleAttention(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        client = MongoClient()
        self.db = client[self.config['db_name']]
        self.split_collection = self.db[self.config['collection_name']]
        self.batch_size = self.config['batch_size']
        self.dict_collection = self.db[self.config['dict_name']]

    def prepare_data(self) -> None:
        print(f"Loading graph dictionary..")
        self.dict = {k["_id"]:
            DirectedData(x=torch.LongTensor(k['graph']["onehot"]),
                         edge_index=torch.LongTensor(k['graph']["edge_index"]),
                         edge_attr=torch.LongTensor(k['graph']["edge_attr"]))
                         for k in tqdm(self.dict_collection.find({}))}

    def custom_collate(self, data):
        conjs = [self.dict[d['conj']] for d in data]
        stmts = [self.dict[d['stmt']] for d in data]
        y = torch.LongTensor([d['y'] for d in data])

        conjs = Batch.from_data_list(conjs)
        stmts = Batch.from_data_list(stmts)

        return conjs, stmts, y

    def setup(self, stage: str):
        if stage == "fit":
            self.train_cursor = self.split_collection.find({"split": "train"}).sort("rand_idx", 1)
            self.train_data = MongoDataset(self.train_cursor, self.config['buf_size'])
            self.train_data = [{'conj': k['conj'], 'stmt': k['stmt'], 'y': k['y']} for k in tqdm(self.train_data)]

            self.val_cursor = self.split_collection.find({"split": "valid"}).sort("rand_idx", 1)
            self.val_data = MongoDataset(self.val_cursor, self.config['buf_size'])
            self.val_data = [{'conj': k['conj'], 'stmt': k['stmt'], 'y': k['y']} for k in tqdm(self.val_data)]

        if stage == "test":
            self.test_cursor = self.split_collection.find({"split": "test"}).sort("rand_idx", 1)
            self.test_data = MongoDataset(self.test_cursor, self.config['buf_size'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate,
                                           num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           collate_fn=self.custom_collate)


# if __name__ == '__main__':
    # conf = {'data_dir': '/home/sean/Documents/phd/repo/aitp/data/utils/holstep_full'}
    # module = H5DataModule(conf)
    #
    # module.setup("fit")
    #
    # count = 0
    # for i, graph in tqdm(enumerate(module.val_dataloader())):
    #     x = graph[0]
    #     count += 1
    #
    # conf = {"db_name": "hol_step",
    #         "collection_name": "train_val_test_data",
    #         "batch_size": 32,
    #         "buf_size": 10000,
    #         "dict_name": "expression_graphs"
    #         }
    # module = MongoDataModuleAttention(conf)
    # module.prepare_data()
    # module.setup("fit")
    #
    # start = time.time()
    # count = 0
    # for i, graph in tqdm(enumerate(module.val_dataloader())):
    #     x = graph[0]
    #     count += 1
    #
    # conf = {
    #     'db_name': 'hol_step',
    #     'collection_name': 'pretrain_graphs',
    #     'batch_size': 32,
    #     'buf_size': 10000,
    #     'options': ['edge_attr', 'edge_index', 'softmax_idx']
    # }
    # module = MongoDataModule(conf)
    # module.setup("fit")
    #
    # start = time.time()
    # count= 0
    # for i, graph in tqdm(enumerate(module.val_dataloader())):
    #     x = graph[0]
    #     count += 1
    #
