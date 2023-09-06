import h5py
import pickle
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index
class PremiseSelectionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def to_pickle(data_list, file):
    with open(file, "wb") as f:
        pickle.dump(data_list, f)

def to_hdf5(data_list: list, h5file: h5py.File, name: str):
    # max_length = max(max(len(data1.x), len(data2.x)) for data1, data2, y in data_list)

    max_length_1 = max(len(data1.x) for data1, data2, y in data_list)
    max_length_2 = max(len(data2.x) for data1, data2, y in data_list)

    # max_edge_length = max(max(len(data1.edge_index[0]), len(data2.edge_index[0])) for data1, data2, y in data_list)

    max_edge_length_1 = max(len(data1.edge_index[0]) for data1, data2, y in data_list)
    max_edge_length_2 = max(len(data2.edge_index[0]) for data1, data2, y in data_list)



    # attention edge index for graph_benchmarks
    max_att_1 = max(len(data1.attention_edge_index[0]) for data1, data2, y in data_list)
    max_att_2 = max(len(data2.attention_edge_index[0]) for data1, data2, y in data_list)

    # len should be constant for ptr
    ptr_len = len(data_list[0][0].ptr)

    x1_dset = h5file.create_dataset(f"{name}/x1", shape=(len(data_list), max_length_1), dtype=np.int64, compression="gzip")#, data_list[0][0].num_node_features))
    edge_index1_dset = h5file.create_dataset(f"{name}/edge_index1", shape=(len(data_list), 2, max_edge_length_1), dtype=np.int64, compression="gzip")
    edge_attr1_dset = h5file.create_dataset(f"{name}/edge_attr1", shape=(len(data_list), max_edge_length_1),dtype=np.int64, compression="gzip")#, data_list[0][0].num_edge_features))
    batch1_dset = h5file.create_dataset(f"{name}/batch1", shape=(len(data_list), max_length_1), dtype=np.int64, compression="gzip")
    ptr1_dset = h5file.create_dataset(f"{name}/ptr1", shape=(len(data_list),ptr_len), dtype=np.int64, compression="gzip")
    edge_ptr1_dset = h5file.create_dataset(f"{name}/edge_ptr1", shape=(len(data_list),ptr_len), dtype=np.int64, compression="gzip")
    attention_edge_1 = h5file.create_dataset(f"{name}/attention_edge1", shape=(len(data_list),2,max_att_1), dtype=np.int64, compression="gzip")

    x2_dset = h5file.create_dataset(f"{name}/x2", shape=(len(data_list), max_length_2), dtype=np.int64, compression="gzip")#, data_list[0][1].num_node_features))
    edge_index2_dset = h5file.create_dataset(f"{name}/edge_index2", shape=(len(data_list), 2, max_edge_length_2), dtype=np.int64, compression="gzip")
    edge_attr2_dset = h5file.create_dataset(f"{name}/edge_attr2", shape=(len(data_list), max_edge_length_2),dtype=np.int64, compression="gzip")#, data_list[0][1].num_edge_features))
    batch2_dset = h5file.create_dataset(f"{name}/batch2", shape=(len(data_list), max_length_2), dtype=np.int64, compression="gzip")
    ptr2_dset = h5file.create_dataset(f"{name}/ptr2", shape=(len(data_list),ptr_len), dtype=np.int64, compression="gzip")
    edge_ptr2_dset = h5file.create_dataset(f"{name}/edge_ptr2", shape=(len(data_list),ptr_len), dtype=np.int64, compression="gzip")
    attention_edge_2 = h5file.create_dataset(f"{name}/attention_edge2", shape=(len(data_list),2,max_att_2), dtype=np.int64, compression="gzip")

    y_dset = h5file.create_dataset(f"{name}/y", shape=(len(data_list),ptr_len-1), dtype=np.int64, compression="gzip")
    
    data_len_1 = h5file.create_dataset(f"{name}/x1_len", shape=(len(data_list),2), dtype=np.int64, compression="gzip")
    data_len_2 = h5file.create_dataset(f"{name}/x2_len", shape=(len(data_list),2), dtype=np.int64, compression="gzip")

    for i, (data1, data2, y) in enumerate(data_list):
        num_nodes1 = data1.num_nodes
        num_edges1 = data1.num_edges
        num_nodes2 = data2.num_nodes
        num_edges2 = data2.num_edges

        x1_dset[i, :num_nodes1] = data1.x.numpy()#.reshape(num_nodes1, 1)

        edge_index1_dset[i, :, :num_edges1] = data1.edge_index.numpy()
        edge_attr1_dset[i, :num_edges1] = data1.edge_attr.numpy()#.reshape(num_edges1, 1)
        batch1_dset[i, :num_nodes1] = data1.batch.numpy()#.reshape(num_nodes1, 1)
        ptr1_dset[i] = data1.ptr.numpy()
        edge_ptr1_dset[i] = np.append([0], data1.softmax_idx.numpy())
        attention_idx = data1.attention_edge_index
        attention_edge_1[i, :, :attention_idx.shape[1]] = attention_idx.numpy()

        x2_dset[i, :num_nodes2] = data2.x.numpy()#.reshape(num_nodes2, 1)
        edge_index2_dset[i, :, :num_edges2] = data2.edge_index.numpy()
        edge_attr2_dset[i, :num_edges2] = data2.edge_attr.numpy()#.reshape(num_edges2, 1)
        batch2_dset[i, :num_nodes2] = data2.batch.numpy()#.reshape(num_nodes2, 1)
        ptr2_dset[i] = data2.ptr.numpy()
        edge_ptr2_dset[i] = np.append([0], data2.softmax_idx.numpy())

        attention_idx = data2.attention_edge_index
        attention_edge_2[i, :, :attention_idx.shape[1]] = attention_idx.numpy()

        y_dset[i] = y.numpy()

        data_len_1[i] = [num_nodes1, num_edges1]

        data_len_2[i] = [num_nodes2, num_edges2]