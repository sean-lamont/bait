import traceback

from torch import nn
import wandb
import random
import cProfile
import models.gnn_edge_labels
from models.graph_transformers.SAT.sat.models import GraphTransformer
import math
import torch
import torch_geometric.utils
from tqdm import tqdm
from models.digae_layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder, SingleLayerDirectedGCNConvEncoder
from models.gnn.digae.digae_model import OneHotDirectedGAE
import json
import models.gnn.formula_net.inner_embedding_network
from torch_geometric.data import Data
import pickle
from data.hol4.ast_def import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


data_dir = "data/hol4/data/"
data_dir = os.path.join(os.getcwd(),data_dir)

with open(data_dir + "dep_data.json") as fp:
    dep_db = json.load(fp)
    
with open(data_dir + "new_db.json") as fp:
    new_db = json.load(fp)

#with open("polished_dict.json") as f:
#    p_d = json.load(f)

# full_db = {}
# count = 0
# for key in new_db.keys():
#     val = new_db[key]
#
#     if key[0] == " ":
#         full_db[key[1:]] = val
#     else:
#         full_db[key] = val
#
# deps = {}
# for key in dep_db.keys():
#     val = dep_db[key]
#
#     if key[0] == " ":
#         deps[key[1:]] = val
#     else:
#         deps[key] = val

with open(data_dir + "torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

# with open("one_hot_dict.pk", "rb") as f:
#     one_hot_dict = pickle.load(f)

with open(data_dir + "train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)

polished_goals = []
for val_ in new_db.values():
    polished_goals.append(val_[2])

tokens = list(set([token.value for polished_goal in polished_goals for token in polished_to_tokens_2(polished_goal)  if token.value[0] != 'V']))

tokens.append("VAR")
tokens.append("VARFUNC")
tokens.append("UNKNOWN")



# Data class for combining two graphs into one for supervised learning (i.e. premise selection). num_nodes_g1 tells us where in data.x to index to get nodes from g1
class CombinedGraphData(Data):
    def __init__(self, combined_x, combined_edge_index, num_nodes_g1, y, combined_edge_attr=None, complete_edge_index=None):
        super().__init__()
        self.y = y
        # node features concatenated along first dimension
        self.x = combined_x
        # adjacency matrix representing nodes from both graphs. Nodes from second graph have num_nodes_g1 added so they represent disjoint sets, but can be computed in parallel
        self.edge_index = combined_edge_index
        self.num_nodes_g1 = num_nodes_g1

        # combined edge features in format as above
        self.combined_edge_attr = combined_edge_attr

        self.complete_edge_index = complete_edge_index



class LinkData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, edge_attr_s=None, edge_attr_t = None,
                 y=None, x_s_one_hot=None, x_t_one_hot=None, edge_index_s_complete=None, edge_index_t_complete=None, depth_x_s=None, depth_x_t=None):
        super().__init__()

        self.edge_index_s = edge_index_s
        self.x_s = x_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t

        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t

        self.x_s_one_hot=x_s_one_hot
        self.x_t_one_hot=x_t_one_hot

        self.edge_index_t_complete = edge_index_t_complete
        self.edge_index_s_complete = edge_index_s_complete

        self.depth_x_s = depth_x_s
        self.depth_x_t = depth_x_t

        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s' or key == 'edge_index_s_complete':
            return self.x_s.size(0)
        elif key == 'edge_index_t' or key == 'edge_index_t_complete':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

# new_train = []
#
# for (x1, x2, y) in train:
#     x1_graph = torch_graph_dict[x1]
#     x2_graph = torch_graph_dict[x2]
#
#     new_train.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))
#
# new_val = []
#
# for (x1, x2, y) in val:
#     x1_graph = torch_graph_dict[x1]
#     x2_graph = torch_graph_dict[x2]
#
#     new_val.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))



# data_dir = "/home/sean/documents/phd/aitp/data/hol4/graph_train_val.pk"
#
#
#
# try:
#     with open(data_dir, "rb") as f:
#         new_train, new_val = pickle.load(f)
# except:
#     print ("generating train/val graphs")
#
#     #edge labelled data
#     new_train = []
#
#     for (x1, x2, y) in train:
#         x1_graph = torch_graph_dict[x1]
#         x2_graph = torch_graph_dict[x2]
#
#         t_nodes = x1_graph.num_nodes
#         s_nodes = x2_graph.num_nodes
#
#         x_t_complete_edge_index = torch.vstack((torch.arange(t_nodes).repeat_interleave(t_nodes), torch.arange(t_nodes).repeat(t_nodes)))
#         x_s_complete_edge_index = torch.vstack((torch.arange(s_nodes).repeat_interleave(s_nodes), torch.arange(s_nodes).repeat(s_nodes)))
#
#
#         # new_train.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))
#
#         new_train.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index))
#
#     new_val = []
#
#     for (x1, x2, y) in val:
#         x1_graph = torch_graph_dict[x1]
#         x2_graph = torch_graph_dict[x2]
#
#         t_nodes = x1_graph.num_nodes
#         s_nodes = x2_graph.num_nodes
#
#         x_t_complete_edge_index = torch.vstack((torch.arange(t_nodes).repeat_interleave(t_nodes), torch.arange(t_nodes).repeat(t_nodes)))
#         x_s_complete_edge_index = torch.vstack((torch.arange(s_nodes).repeat_interleave(s_nodes), torch.arange(s_nodes).repeat(s_nodes)))
#
#
#         new_val.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index))
#
#     with open(data_dir, "wb") as f:
#         pickle.dump((new_train, new_val), f)
#
#

# data_dir = "combined_graphs_train_val.pk"
#
#
#
# try:
#     with open(data_dir, "rb") as f:
#         new_train, new_val = pickle.load(f)
# except:
#     print ("generating train/val graphs")
#
#     #edge labelled data
#     new_train = []
#
#     for (x1, x2, y) in train:
#         x1_graph = torch_graph_dict[x1]
#         x2_graph = torch_graph_dict[x2]
#
#         t_nodes = x1_graph.num_nodes
#         s_nodes = x2_graph.num_nodes
#         total_nodes = t_nodes + s_nodes
#
#         # x_t_complete_edge_index = torch.vstack((torch.arange(total_nodes).repeat_interleave(total_nodes), torch.arange(total_nodes).repeat(total_nodes)))
#
#         combined_nodes = torch.cat([x1_graph.x, x2_graph.x],dim=0)
#
#         combined_index = torch.cat([x1_graph.edge_index, x2_graph.edge_index + (torch.ones(x2_graph.edge_index.shape).long() * t_nodes)], dim=1)
#
#         new_train.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index))
#
#     new_val = []
#
#     for (x1, x2, y) in val:
#         x1_graph = torch_graph_dict[x1]
#         x2_graph = torch_graph_dict[x2]
#
#         t_nodes = x1_graph.num_nodes
#         s_nodes = x2_graph.num_nodes
#
#         x_t_complete_edge_index = torch.vstack((torch.arange(t_nodes).repeat_interleave(t_nodes), torch.arange(t_nodes).repeat(t_nodes)))
#         x_s_complete_edge_index = torch.vstack((torch.arange(s_nodes).repeat_interleave(s_nodes), torch.arange(s_nodes).repeat(s_nodes)))
#
#
#         new_val.append(linkdata(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index))
#
#     with open(data_dir, "wb") as f:
#         pickle.dump((new_train, new_val), f)
# #


# todo combined data
#
# tmp = []
# for i, lnk_data in enumerate(new_train):
#     xt_nodes = lnk_data.x_t
#     xs_nodes = lnk_data.x_s
#
#
#     xt_edge_index = lnk_data.edge_index_t
#     xs_edge_index = lnk_data.edge_index_s
#
#     y = lnk_data.y
#
#     t_nodes = xt_nodes.shape[0]
#     s_nodes = xs_nodes.shape[0]
#
#     total_nodes = t_nodes + s_nodes
#
#     # x_t_complete_edge_index = torch.vstack((torch.arange(total_nodes).repeat_interleave(total_nodes), torch.arange(total_nodes).repeat(total_nodes)))
#
#     combined_nodes = torch.cat([xt_nodes, xs_nodes],dim=0)
#
#     combined_index = torch.cat([xt_edge_index, xs_edge_index + (torch.ones(xs_edge_index.shape).long() * t_nodes)], dim=1)
#
#     if hasattr(lnk_data, 'edge_attr_t'):
#         xt_edge_attr = lnk_data.edge_attr_t
#         xs_edge_attr = lnk_data.edge_attr_s
#         combined_attr = torch.cat([xt_edge_attr, xs_edge_attr], dim=0)
#         tmp.append(combinedgraphdata(combined_x = combined_nodes, combined_edge_index=combined_index, combined_edge_attr = combined_attr, y=y,num_nodes_g1=t_nodes))
#
#     else:
#         tmp.append(combinedgraphdata(combined_x = combined_nodes, combined_edge_index=combined_index, num_nodes_g1=t_nodes))
#
# new_train = tmp
#

def get_directed_edge_index(num_nodes, edge_idx):
    from_idx = []
    to_idx = []

    for i in range(0,num_nodes-1):
        # to_idx = [i]
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx)
            # print (f"ancestor nodes for {i}: {ancestor_nodes}")
        except:
            print (f"exception {i, num_nodes, edge_idx}")

        # ancestor_nodes = ancestor_nodes.item()
        found_nodes = list(ancestor_nodes.numpy())
        found_nodes.remove(i)


        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx, flow='target_to_source')

        found_nodes = list(children_nodes.numpy())
        found_nodes.remove(i)
        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


# def get_directed_edge_index(num_nodes, edge_idx):
#     from_idx = []
#     to_idx = []
#
#     for i in range(0,num_nodes-1):
#         # to_idx = [i]
#         try:
#             ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx)
#         except:
#             print (f"exception {i, num_nodes, edge_idx}")
#
#         # ancestor_nodes = ancestor_nodes.item()
#         found_nodes = list(ancestor_nodes).remove(i)
#
#         if found_nodes is not none:
#             for node in found_nodes:
#                 to_idx.append(i)
#                 from_idx.append(node)
#
#         children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx, flow='target_to_source')
#         # children_nodes = children_nodes.item()
#         # print (found_nodes, children_nodes, i, self_idx.item(), edge_idx)
#         found_nodes = list(children_nodes).remove(i)
#
#         if found_nodes is not none:
#             for node in found_nodes:
#                 to_idx.append(i)
#                 from_idx.append(node)
#
#     return torch.tensor([from_idx, to_idx], dtype=torch.long)

# probably slow, could recursively do k-hop subgraph with k = 1 instead
def get_depth_from_graph(num_nodes, edge_index):
    from_idx = edge_index[0]
    to_idx = edge_index[1]


    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes , _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i, edge_index=edge_index, flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes


    return depths


# data_dir = "graph_train_val_directed.pk"
#
# try:
#     with open(data_dir, "rb") as f:
#         new_train, new_val = pickle.load(f)
#
# except:
#     print ("Generating train/val graphs with directed edge index for attention propagation")
#
#     new_graph_dict = {}
#
#     for k,graph in tqdm(torch_graph_dict.items()):
#         complete_edge_index = get_directed_edge_index(graph.num_nodes, graph.edge_index)
#         depth = get_depth_from_graph(graph.num_nodes, graph.edge_index)
#         new_graph_dict[k] = (graph, complete_edge_index, depth)
#
#     #edge labelled data
#     new_train = []
#
#     for (x1, x2, y) in tqdm(train):
#         x1_graph, x_t_complete_edge_index, depth_x_t = new_graph_dict[x1]
#         x2_graph, x_s_complete_edge_index, depth_x_s = new_graph_dict[x2]
#
#         new_train.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index, depth_x_s=depth_x_s, depth_x_t=depth_x_t))
#
#     new_val = []
#
#     for (x1, x2, y) in tqdm(val):
#
#         x1_graph, x_t_complete_edge_index, depth_x_t = new_graph_dict[x1]
#         x2_graph, x_s_complete_edge_index, depth_x_s = new_graph_dict[x2]
#
#         new_val.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1], edge_index_t_complete=x_t_complete_edge_index, edge_index_s_complete=x_s_complete_edge_index, depth_x_s=depth_x_s, depth_x_t=depth_x_t))
#
#
#     with open(data_dir, "wb") as f:
#         pickle.dump((new_train, new_val), f)









# old functions
# def loss(graph_net, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):
#
#     g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     eps = 1e-6
#
#     preds = torch.clip(preds, eps, 1-eps)
#
#     return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# from tqdm import tqdm
#
# def accuracy(graph_net, batch, fc):
#
#     g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device),batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device),batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds>0.5).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
#
#
# def run(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)
#
#     optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     training_losses = []
#
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#
#             optimiser_fc.zero_grad()
#
#             optimiser_gnn.zero_grad()
#
#             loss_val = loss(graph_net, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#             loss_val.backward()
#
#             optimiser_fc.step()
#
#             optimiser_gnn.step()
#
#             training_losses.append(loss_val.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 validation_loss = accuracy(graph_net, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#                 val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#     #only save encoder for now
#     if save == True:
#         torch.save(graph_net, "model_checkpoints/gnn_encoder_latest")
#
#
#     return training_losses, val_losses
#
# def plot_losses(train_loss, val_loss):
#
#     #plt.plot(np.convolve([t[0].cpu().numpy() for t in val_loss], np.ones(40)/40, mode='valid'))
#     plt.plot([t[0].cpu().numpy() for t in val_loss])
#     plt.show()
#     plt.plot(np.convolve([t.cpu().numpy() for t in train_loss], np.ones(1000)/1000, mode='valid'))
#     plt.show()
#
#
# #define setup for separate premise and goal GNNs
#
# def loss_2(graph_net_1, graph_net_2, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):
#
#     g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     eps = 1e-6
#
#     preds = torch.clip(preds, eps, 1-eps)
#
#     return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# from tqdm import tqdm
#
# def accuracy_2(graph_net_1, graph_net_2, batch, fc):
#
#     g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device),batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device),batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds>0.5).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
#
#
# def run_2(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net_1 = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     graph_net_2 = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     fc = inner_embedding_network.F_c_module_(embedding_dim * 2).to(device)
#
#     optimiser_gnn_1 = torch.optim.Adam(list(graph_net_1.parameters()), lr=step_size, weight_decay=decay_rate)
#     optimiser_gnn_2 = torch.optim.Adam(list(graph_net_2.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     training_losses = []
#
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#
#             optimiser_fc.zero_grad()
#
#             optimiser_gnn_1.zero_grad()
#             optimiser_gnn_2.zero_grad()
#
#             loss_val = loss_2(graph_net_1,graph_net_2, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#             loss_val.backward()
#
#             optimiser_gnn_1.step()
#             optimiser_gnn_2.step()
#
#             optimiser_fc.step()
#
#
#             training_losses.append(loss_val.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 validation_loss = accuracy_2(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#                 val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#     #only save encoder for now
#     if save == True:
#         torch.save(graph_net_1, "model_checkpoints/gnn_encoder_latest_1")
#         torch.save(graph_net_2, "model_checkpoints/gnn_encoder_latest_2")
#
#
#     return training_losses, val_losses
#
# # run_2(1e-3, 0, 20, 1024, 256, 4, False)
#
#
# #define setup for separate premise and goal GNNs
#
# def loss_edges(graph_net_1, graph_net_2, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):
#
#     g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device), batch.edge_attr_t.to(device), batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.edge_attr_s.to(device), batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     eps = 1e-6
#
#     preds = torch.clip(preds, eps, 1-eps)
#
#     return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# from tqdm import tqdm
#
# def accuracy_edges(graph_net_1, graph_net_2, batch, fc):
#
#     g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device),batch.edge_attr_t.to(device), batch.x_t_batch.to(device))
#
#     g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.edge_attr_s.to(device), batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds>0.5).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
#
#
# def run_edges(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
#     graph_net_1 = models.gnn_edge_labels.message_passing_gnn_edges(len(tokens), embedding_dim, graph_iterations, device)
#
#     graph_net_2 = models.gnn_edge_labels.message_passing_gnn_edges(len(tokens), embedding_dim, graph_iterations, device)
#
#     # graph_net_1 = gnn_edge_labels.message_passing_gnn_edges_gine(len(tokens), embedding_dim, graph_iterations, device)
#     #
#     # graph_net_2 = gnn_edge_labels.message_passing_gnn_edges_gine(len(tokens), embedding_dim, graph_iterations, device)
#
#     fc = models.gnn_edge_labels.F_c_module_(embedding_dim * 2).to(device)
#
#     optimiser_gnn_1 = torch.optim.Adam(list(graph_net_1.parameters()), lr=step_size, weight_decay=decay_rate)
#     optimiser_gnn_2 = torch.optim.Adam(list(graph_net_2.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     training_losses = []
#
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#
#             optimiser_fc.zero_grad()
#
#             optimiser_gnn_1.zero_grad()
#             optimiser_gnn_2.zero_grad()
#
#             loss_val = loss_edges(graph_net_1,graph_net_2, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#             loss_val.backward()
#
#             optimiser_gnn_1.step()
#             optimiser_gnn_2.step()
#
#             optimiser_fc.step()
#
#
#             training_losses.append(loss_val.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 validation_loss = accuracy_edges(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#                 val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#     #only save encoder for now
#     if save == True:
#         torch.save(graph_net_1, "model_checkpoints/gnn_encoder_latest_1")
#         torch.save(graph_net_2, "model_checkpoints/gnn_encoder_latest_2")
#
#
#     return training_losses, val_losses
#
#
# # run_2(1e-3, 0, 20, 1024, 256, 4, False)
#
# def accuracy_digae(model_1, model_2, batch, fc):
#
#     data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device))
#     data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device))
#
#     # g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))
#     #
#     # g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))
#
#     u1 = data_1.x.clone().to(device)
#     v1 = data_1.x.clone().to(device)
#
#     train_pos_edge_index_1 = data_1.edge_index.clone().to(device)
#
#     u2 = data_2.x.clone().to(device)
#     v2 = data_2.x.clone().to(device)
#
#     train_pos_edge_index_2 = data_2.edge_index.clone().to(device)
#
#     graph_enc_1 = model_1.encode_and_pool(u1, v1, train_pos_edge_index_1, batch.x_t_batch.to(device))
#
#     graph_enc_2 = model_2.encode_and_pool(u2, v2, train_pos_edge_index_2, batch.x_s_batch.to(device))
#
#     preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds>0.5).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def run_digae(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=batch_size, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     hidden_dim = 64
#     out_dim = 64
#
#     initial_encoder = inner_embedding_network.F_x_module_(len(tokens), embedding_dim).to(device)
#     decoder = DirectedInnerProductDecoder()
#
#     # encoder = DirectedGCNConvEncoder(embedding_dim, hidden_dim, out_dim, alpha=0.2, beta=0.8,
#     #                                         self_loops=True,
#     #                                         adaptive=False)
#
#     graph_net_1 = OneHotDirectedGAE(initial_encoder, embedding_dim, hidden_dim, out_dim).to(device)
#     graph_net_2 = OneHotDirectedGAE(initial_encoder, embedding_dim, hidden_dim, out_dim).to(device)
#
#     fc = gnn_edge_labels.F_c_module_(embedding_dim * 4).to(device)
#
#     # op_enc =torch.optim.Adam(encoder.parameters(), lr=step_size)
#     op_g1 =torch.optim.Adam(graph_net_1.parameters(), lr=step_size)
#     op_g2 =torch.optim.Adam(graph_net_2.parameters(), lr=step_size)
#     op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)
#
#     training_losses = []
#
#     val_losses = []
#     best_acc = 0.
#
#     for j in range(num_epochs):
#         print (f"Epoch: {j}")
#         for i, batch in tqdm(enumerate(loader)):
#
#             # op_enc.zero_grad()
#             op_g1.zero_grad()
#             op_g2.zero_grad()
#             op_fc.zero_grad()
#
#             data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device))
#             data_2 = Data(x = batch.x_s.to(device), edge_index = batch.edge_index_s.to(device))
#
#
#             u1 = data_1.x.clone().to(device)
#             v1 = data_1.x.clone().to(device)
#
#
#             train_pos_edge_index_1 = data_1.edge_index.clone().to(device)
#
#             u2 = data_2.x.clone().to(device)
#             v2 = data_2.x.clone().to(device)
#
#             train_pos_edge_index_2 = data_2.edge_index.clone().to(device)
#
#             graph_enc_1 = graph_net_1.encode_and_pool(u1, v1, train_pos_edge_index_1, batch.x_t_batch.to(device))
#
#             graph_enc_2 = graph_net_2.encode_and_pool(u2, v2, train_pos_edge_index_2, batch.x_s_batch.to(device))
#
#
#
#
#             # print (graph_enc_1.shape, graph_enc_2.shape)
#             preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
#
#             eps = 1e-6
#
#             preds = torch.clip(preds, eps, 1 - eps)
#
#             loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
#
#             loss.backward()
#
#             # op_enc.step()
#             op_g1.step()
#             op_g2.step()
#             op_fc.step()
#
#             training_losses.append(loss.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 validation_loss = accuracy_digae(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#                 val_loader = iter(DataLoader(new_val, batch_size=batch_size, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#                 if validation_loss > best_acc:
#                     best_acc = validation_loss
#                     print (f"New best validation accuracy: {best_acc}")
#                     #only save encoder if best accuracy so far
#                     if save == True:
#                         torch.save(graph_net_1, "model_checkpoints/gnn_encoder_bow_goal")
#                         torch.save(graph_net_2, "model_checkpoints/gnn_encoder_bow_premise")
#
#     print (f"Best validation accuracy: {best_acc}")
#
#     return training_losses, val_losses



# print ("running digae") # run_digae(1e-3, 0, 200, 1024, 64, 2, save=False)


def positional_encoding(d_model, depth_vec):

    size, _ = depth_vec.shape

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(size, d_model)

    pe[:, 0::2] = torch.sin(depth_vec * div_term)
    pe[:, 1::2] = torch.cos(depth_vec * div_term)

    return pe


# def ptr_to_complete_edge_index(ptr):
#     from_lists = [torch.arange(ptr[i], ptr[i+1]).repeat_interleave(ptr[i+1] - ptr[i]) for i in range(len(ptr) - 1)]
#     to_lists = [torch.arange(ptr[i], ptr[i+1]).repeat(ptr[i+1] - ptr[i]) for i in range(len(ptr) - 1)]
#     combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
#     return combined_complete_edge_index

# def to_batch(list_data, data_dict):
#
#     batch_list = []
#     for (y, conj, stmt) in list_data:
#
#         #x1/x_t is conj, x2/x_s is stmt
#
#         x, edge_index, _, depth = data_dict[conj]
#
#         x1_mat = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=1909)#.float()
#
#         x1_edge_index = torch.LongTensor(edge_index)
#         # x1_complete_edge_index = torch.LongTensor(complete_edge_index)
#         x1_depth = torch.LongTensor(depth)
#
#         x, edge_index, _, depth = data_dict[stmt]
#
#         x2_mat = torch.nn.functional.one_hot(torch.LongTensor(x), num_classes=1909)#.float()
#
#         x2_edge_index = torch.LongTensor(edge_index)
#         # x2_complete_edge_index = torch.LongTensor(complete_edge_index)
#         x2_depth = torch.LongTensor(depth)
#
#         batch_list.append(LinkData(edge_index_s=x2_edge_index, x_s=x2_mat, edge_index_t=x1_edge_index, x_t=x1_mat,  y=torch.tensor(y), depth_x_s=x2_depth, depth_x_t=x1_depth))
#
#     loader = iter(DataLoader(batch_list, batch_size=len(batch_list), follow_batch=['x_s', 'x_t']))
#
#     batch = next(iter(loader))
#
#     batch.edge_index_t_complete = ptr_to_complete_edge_index(batch.x_t_ptr)
#     batch.edge_index_s_complete = ptr_to_complete_edge_index(batch.x_s_ptr)
#
#     return batch



# def run_sat(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=batch_size, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     hidden_dim = 64
#     out_dim = 64
#
#     # graph_net1 = GraphTransformer(len(tokens), 2, embedding_dim, dim_feedforward=256, num_heads=2, num_layers=2, in_embed=False, se='formula-net', abs_pe=True, abs_pe_dim=embedding_dim,  use_edge_attr=True, num_edge_features=20).to(device)
#     # graph_net2 = GraphTransformer(len(tokens), 2, embedding_dim, dim_feedforward=256, num_heads=2, num_layers=2, in_embed=False, se='formula-net', abs_pe=True, abs_pe_dim=embedding_dim,  use_edge_attr=True, num_edge_features=20).to(device)
#
#     graph_net1 = GraphTransformer(len(tokens), 2, embedding_dim, dim_feedforward=256, num_heads=4, num_layers=4, in_embed=False, se='formula-net', abs_pe=True, abs_pe_dim=embedding_dim, k_hop=graph_iterations).to(device)
#     graph_net2 = GraphTransformer(len(tokens), 2, embedding_dim, dim_feedforward=256, num_heads=4, num_layers=4, in_embed=False, se='formula-net', abs_pe=True, abs_pe_dim=embedding_dim, k_hop=graph_iterations).to(device)
#
#
#     fc = models.gnn_edge_labels.F_c_module_(embedding_dim * 2).to(device)
#
#     # op_enc =torch.optim.Adam(encoder.parameters(), lr=step_size)
#     op_g1 =torch.optim.Adam(graph_net1.parameters(), lr=step_size)
#     op_g2 =torch.optim.Adam(graph_net2.parameters(), lr=step_size)
#     op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)
#
#     training_losses = []
#
#     val_losses = []
#     best_acc = 0.
#
#     for j in range(num_epochs):
#         print (f"Epoch: {j}")
#         for i, batch in tqdm(enumerate(loader)):
#
#             # op_enc.zero_grad()
#             op_g1.zero_grad()
#             op_g2.zero_grad()
#             op_fc.zero_grad()
#
#
#             complete_edge_index_t = ptr_to_complete_edge_index(batch.x_t_ptr)
#             complete_edge_index_s = ptr_to_complete_edge_index(batch.x_s_ptr)
#
#             # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_t_complete.to(device), abs_pe =positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_t.long().to(device))
#             # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_s_complete.to(device), abs_pe = positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_s.long().to(device))
#
#             # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = batch.x_t_ptr, complete_edge_index = complete_edge_index_t, abs_pe =positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_t.long().to(device))
#             # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = batch.x_s_ptr, complete_edge_index = complete_edge_index_s, abs_pe = positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_s.long().to(device))
#
#             # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = batch.x_t_ptr, complete_edge_index = complete_edge_index_t, abs_pe =positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_t.long().to(device))
#             # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = batch.x_s_ptr, complete_edge_index = complete_edge_index_s, abs_pe = positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_s.long().to(device))
#
#             data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = batch.x_t_ptr.to(device), complete_edge_index = complete_edge_index_t.to(device),  edge_attr = batch.edge_attr_t.long().to(device))
#             data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = batch.x_s_ptr.to(device), complete_edge_index = complete_edge_index_s.to(device),  edge_attr = batch.edge_attr_s.long().to(device))
#
#             graph_enc_1 = graph_net1(data_1)
#             graph_enc_2 = graph_net2(data_2)
#
#             preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
#
#             eps = 1e-6
#
#             preds = torch.clip(preds, eps, 1 - eps)
#
#             loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
#
#             loss.backward()
#
#             # op_enc.step()
#             op_g1.step()
#             op_g2.step()
#             op_fc.step()
#
#             training_losses.append(loss.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 validation_loss = accuracy_sat(graph_net1, graph_net2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#                 random.shuffle(new_val)
#                 val_loader = iter(DataLoader(new_val, batch_size=16, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#                 if validation_loss > best_acc:
#                     best_acc = validation_loss
#                     print (f"New best validation accuracy: {best_acc}")
#                     #only save encoder if best accuracy so far
#                     if save == True:
#                         torch.save(graph_net1, "model_checkpoints/gnn_transformer_goal")
#                         torch.save(graph_net2, "model_checkpoints/gnn_transformer_premise")
#
#     print (f"Best validation accuracy: {best_acc}")
#
#     return training_losses, val_losses
#
#
# def accuracy_sat(model_1, model_2, batch, fc):
#     # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_t_complete.to(device), abs_pe =positional_encoding(128, batch.depth_x_t.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_t.long().to(device))
#     # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device), complete_edge_index = batch.edge_index_s_complete.to(device), abs_pe = positional_encoding(128, batch.depth_x_s.unsqueeze(1)).to(device), edge_attr = batch.edge_attr_s.long().to(device))
#
#     complete_edge_index_t = ptr_to_complete_edge_index(batch.x_t_ptr)
#     complete_edge_index_s = ptr_to_complete_edge_index(batch.x_s_ptr)
#
#
#     data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = batch.x_t_ptr.to(device), complete_edge_index = complete_edge_index_t.to(device),  edge_attr = batch.edge_attr_t.long().to(device))
#     data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = batch.x_s_ptr.to(device), complete_edge_index = complete_edge_index_s.to(device),  edge_attr = batch.edge_attr_s.long().to(device))
#
#
#     # data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device), batch = batch.x_t_batch.to(device), ptr = batch.x_t_ptr, complete_edge_index = complete_edge_index_t,  edge_attr = batch.edge_attr_t.long().to(device))
#     # data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device), batch = batch.x_s_batch.to(device), ptr = batch.x_s_ptr, complete_edge_index = complete_edge_index_s,  edge_attr = batch.edge_attr_s.long().to(device))
#
#     graph_enc_1 = model_1(data_1)
#
#     graph_enc_2 = model_2(data_2)
#
#     preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))
#
#     preds = torch.flatten(preds)
#
#     preds = (preds>0.5).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
#

# run_edges(1e-3, 0, 200, 1024, 256, 0, False)


# cProfile.run("run_sat(1e-3, 0, 40, batch_size=1, embedding_dim=64, graph_iterations=0, save=False)", sort='cumtime')
#
# run_sat(1e-4, 0, 200, batch_size=32, embedding_dim=128, graph_iterations=4, save=False)

def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

#todo remove this when cartesian product used
def ptr_to_complete_edge_index(ptr):
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index


def to_batch(list_data, data_dict, directed_attention=False):
    batch_list = []
    for (x1, x2, y) in list_data:
        # x1/x_t is conj, x2/x_s is stmt
        conj = x1
        stmt = x2

        conj_graph = data_dict[conj]
        stmt_graph = data_dict[stmt]

        if directed_attention:
            # with directed edge index and depth
            batch_list.append(
                LinkData(edge_index_s=stmt_graph.edge_index, x_s=stmt_graph.x, edge_attr_s=stmt_graph.edge_attr,
                         edge_index_s_complete=stmt_graph.complete_edge_index, depth_x_s=stmt_graph.depth,
                         edge_index_t=conj_graph.edge_index, x_t=conj_graph.x, edge_attr_t=conj_graph.edge_attr,
                         edge_index_t_complete=conj_graph.complete_edge_index, depth_x_t=conj_graph.depth,
                         y=torch.tensor(y)))
        else:

            if hasattr(conj_graph, 'depth') and hasattr(stmt_graph, 'depth'):
                depth_x_s = stmt_graph.depth
                depth_x_t = conj_graph.depth
            else:
                depth_x_s = depth_x_t = None
            batch_list.append(LinkData(edge_index_s=stmt_graph.edge_index, x_s=stmt_graph.x, edge_attr_s=stmt_graph.edge_attr, depth_x_s=depth_x_s, edge_index_t=conj_graph.edge_index, x_t=conj_graph.x, edge_attr_t=conj_graph.edge_attr, depth_x_t = depth_x_t, y=torch.tensor(y)))



    loader = iter(DataLoader(batch_list, batch_size=len(batch_list), follow_batch=['x_s', 'x_t']))

    batch = next(iter(loader))

    # todo do this with cartesian product as in combined experiment
    if not directed_attention:
        # for complete edge index non directed
        batch.edge_index_t_complete = ptr_to_complete_edge_index(batch.x_t_ptr)
        batch.edge_index_s_complete = ptr_to_complete_edge_index(batch.x_s_ptr)

    return batch


def get_model(config):

    if config['model_type'] == 'graph_benchmarks':
        return GraphTransformer(in_size=config['vocab_size'],
                                num_class=2,
                                d_model=config['embedding_dim'],
                                dim_feedforward=config['dim_feedforward'],
                                num_heads=config['num_heads'],
                                num_layers=config['num_layers'],
                                in_embed=config['in_embed'],
                                se=config['se'],
                                abs_pe=config['abs_pe'],
                                abs_pe_dim=config['abs_pe_dim'],
                                use_edge_attr=config['use_edge_attr'],
                                num_edge_features=200,
                                dropout=config['dropout'],
                                k_hop=config['gnn_layers'])

    elif config['model_type'] == 'formula-net':
        return models.gnn.formula_net.inner_embedding_network.FormulaNet(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'formula-net-edges':
        return models.gnn_edge_labels.FormulaNetEdges(config['vocab_size'], config['embedding_dim'], config['gnn_layers'])

    elif config['model_type'] == 'digae':
        return None

    elif config['model_type'] == 'classifier':
        return None

    else:
        return None


train_data = train
val_data = val


# for k,g in tqdm(torch_graph_dict.items()):
#     complete_edge_index = get_directed_edge_index(g.num_nodes, g.edge_index)
#     g.complete_edge_index = complete_edge_index
#     torch_graph_dict[k] = g
#
# with open("torch_graph_dict_directed.pk", "wb") as f:
#     pickle.dump(torch_graph_dict, f)


# todo move generation of this to data module
# with open("/home/sean/Documents/phd/aitp/sat/hol4/supervised/torch_graph_dict_directed_depth.pk", "rb") as f:
#     torch_graph_dict = pickle.load(f)


# def run_dual_encoders(model_config, exp_config):
def run_dual_encoders(config):

    model_config = config['model_config']
    exp_config = config['exp_config']

    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_net_1 = get_model(model_config).to(device)
    graph_net_2 = get_model(model_config).to(device)

    # graph_net_1 = nn.DataParallel(graph_net_1).to(device)
    # graph_net_2 = nn.DataParallel(graph_net_2).to(device)

    # graph_net_1 = torch.compile(graph_net_1)
    # graph_net_2 = torch.compile(graph_net_2)

    print ("Model details:")

    print(graph_net_1)


    embedding_dim = model_config['embedding_dim']
    lr = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    save = exp_config['model_save']
    val_size = exp_config['val_size']
    logging = exp_config['logging']

    if 'directed_attention' in model_config:
        directed_attention = model_config['directed_attention']
    else:
        directed_attention = False

    if logging:
        wandb.log({"Num_model_params": sum([p.numel() for p in graph_net_1.parameters() if p.requires_grad])})

    #wandb load
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']

    fc = models.gnn_edge_labels.BinaryClassifier(embedding_dim * 2).to(device)
    # fc = nn.DataParallel(fc).to(device)
    # fc = torch.compile(fc)

    op_g1 = torch.optim.AdamW(graph_net_1.parameters(), lr=lr, weight_decay=weight_decay)
    op_g2 = torch.optim.AdamW(graph_net_2.parameters(), lr=lr, weight_decay=weight_decay)
    op_fc = torch.optim.AdamW(fc.parameters(), lr=lr, weight_decay=weight_decay)

    training_losses = []

    val_losses = []
    best_acc = 0.

    inds = list(range(0, len(train_data), batch_size))
    inds.append(len(train_data) - 1)

    random.shuffle(train_data)

    for j in range(epochs):
        print(f"Epoch: {j}")
        # for i, batch in tqdm(enumerate(loader)):
        err_count = 0
        for i in tqdm(range(0, len(inds) - 1)):

            from_idx = inds[i]
            to_idx = inds[i + 1]

            try:
                batch = to_batch(train_data[from_idx:to_idx], torch_graph_dict, directed_attention=directed_attention)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

            # op_enc.zero_grad()
            op_g1.zero_grad()
            op_g2.zero_grad()
            op_fc.zero_grad()


            data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
                          batch=batch.x_t_batch.to(device),
                          ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
                          complete_edge_index=batch.edge_index_t_complete.to(device),
                          edge_attr=batch.edge_attr_t.long().to(device))

            data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
                          batch=batch.x_s_batch.to(device),
                          ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
                          complete_edge_index=batch.edge_index_s_complete.to(device),
                          edge_attr=batch.edge_attr_s.long().to(device))


            # data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
            #               batch=batch.x_t_batch.to(device),
            #               ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
            #               complete_edge_index=batch.edge_index_t_complete.to(device),
            #               edge_attr=batch.edge_attr_t.long().to(device),
            #           abs_pe=positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device))
            #
            # data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
            #               batch=batch.x_s_batch.to(device),
            #               ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
            #               complete_edge_index=batch.edge_index_s_complete.to(device),
            #               edge_attr=batch.edge_attr_s.long().to(device),
            #               abs_pe=positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device))
            #

            try:
                graph_enc_1 = graph_net_1(data_1)
                print (f"data1 {data_1}")

                graph_enc_2 = graph_net_2(data_2)

                preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

                eps = 1e-6

                preds = torch.clip(preds, eps, 1 - eps)

                loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))


                loss.backward()

                # op_enc.step()
                op_g1.step()
                op_g2.step()
                op_fc.step()


            except Exception as e:
                err_count += 1
                if err_count > 100:
                    return Exception("Too many errors in training")
                print(f"Error in training {e}")
                traceback.print_exc()
                continue

            training_losses.append(loss.detach() / batch_size)

            if i % (10000 // batch_size) == 0:

                graph_net_1.eval()
                graph_net_2.eval()

                val_count = []

                random.shuffle(val_data)

                val_inds = list(range(0, len(val_data), batch_size))
                val_inds.append(len(val_data) - 1)


                for k in tqdm(range(0, val_size // batch_size)):
                    val_err_count = 0

                    from_idx_val = val_inds[k]
                    to_idx_val = val_inds[k + 1]

                    try:
                        val_batch = to_batch(val_data[from_idx_val:to_idx_val], torch_graph_dict, directed_attention)
                        validation_loss = val_acc_dual_encoder(graph_net_1, graph_net_2, val_batch,
                                                               fc, embedding_dim, directed_attention)  # , fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
                        val_count.append(validation_loss.detach())
                    except Exception as e:
                        # print(f"Error {e}, batch: {val_batch}")
                        print (e)
                        val_err_count += 1
                        continue

                print (f"Val errors: {val_err_count}")

                validation_loss = (sum(val_count) / len(val_count)).detach()
                val_losses.append((validation_loss, j, i))

                print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print("Val acc: {}".format(validation_loss.detach()))

                print (f"Failed batches: {err_count}")

                if logging:
                    wandb.log({"acc": validation_loss.detach(),
                               "train_loss_avg": sum(training_losses[-100:]) / len(training_losses[-100:]),
                               "epoch": j})

                if validation_loss > best_acc:
                    best_acc = validation_loss
                    print(f"New best validation accuracy: {best_acc}")
                    # only save encoder if best accuracy so far
                    if save == True:
                        torch.save(graph_net_1, exp_config['model_dir'] + "/gnn_transformer_goal_hol4")
                        torch.save(graph_net_2, exp_config['model_dir'] + "/gnn_transformer_premise_hol4")

                    # wandb save
                    # torch.save({  # Save our checkpoint loc
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss,
                    # }, CHECKPOINT_PATH)
                    # wandb.save(CHECKPOINT_PATH)  # saves c


                graph_net_1.train()
                graph_net_2.train()

        if logging:
            wandb.log({"failed_batches": err_count})

    print(f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses


def val_acc_dual_encoder(model_1, model_2, batch, fc, embedding_dim, directed_attention):




    data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
                  batch=batch.x_t_batch.to(device),
                  ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
                  complete_edge_index=batch.edge_index_t_complete.to(device),
                  edge_attr=batch.edge_attr_t.long().to(device))

    data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
                  batch=batch.x_s_batch.to(device),
                  ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
                  complete_edge_index=batch.edge_index_s_complete.to(device),
                  edge_attr=batch.edge_attr_s.long().to(device))



    # if directed_attention:
    # data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
    #               batch=batch.x_t_batch.to(device),
    #               ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
    #               complete_edge_index=batch.edge_index_t_complete.to(device),
    #               edge_attr=batch.edge_attr_t.long().to(device),
    #             abs_pe = positional_encoding(embedding_dim, batch.depth_x_t.unsqueeze(1)).to(device))
    #
    # data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
    #             batch = batch.x_s_batch.to(device),
    #             ptr = torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
    #             complete_edge_index = batch.edge_index_s_complete.to(device),
    #             edge_attr = batch.edge_attr_s.long().to(device),
    #             abs_pe = positional_encoding(embedding_dim, batch.depth_x_s.unsqueeze(1)).to(device))
    #
    # else:
    #     data_1 = Data(x=batch.x_t.float().to(device), edge_index=batch.edge_index_t.to(device),
    #                   batch=batch.x_t_batch.to(device),
    #                   ptr=torch._convert_indices_from_coo_to_csr(batch.x_t_batch, len(batch)).to(device),
    #                   complete_edge_index=batch.edge_index_t_complete.to(device),
    #                   edge_attr=batch.edge_attr_t.to(device))#,
    #
    #     data_2 = Data(x=batch.x_s.float().to(device), edge_index=batch.edge_index_s.to(device),
    #                   batch=batch.x_s_batch.to(device),
    #                   ptr=torch._convert_indices_from_coo_to_csr(batch.x_s_batch, len(batch)).to(device),
    #                   complete_edge_index=batch.edge_index_s_complete.to(device),
    #                   edge_attr=batch.edge_attr_s.to(device))#,

    graph_enc_1 = model_1(data_1)

    graph_enc_2 = model_2(data_2)

    preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds > 0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)

sat_config = {
    "model_type": "graph_benchmarks",
    "vocab_size": len(tokens),
    "embedding_dim": 128,
    "dim_feedforward": 256,
    "num_heads": 8,
    "num_layers": 4,
    "in_embed": False,
    "se": "pna",
    "abs_pe": False,
    "abs_pe_dim": 256,
    "use_edge_attr": False,
    "dropout": 0.2,
    "gnn_layers": 4,
    "directed_attention": False,
}

exp_config = {
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-6,
    "batch_size": 128,
    "model_save": True,
    "val_size": 2048,
    "logging": False,
    "model_dir": "/home/sean/Documents/phd/aitp/sat/hol4/supervised/model_checkpoints"
}

formula_net_config = {
    "model_type": "formula-net",
    "vocab_size": len(tokens),
    "embedding_dim": 256,
    "gnn_layers": 4,
}


def main():
    wandb.init(
        project="hol4_premise_selection",

        name="Directed Attention Sweep Separate Encoder",
        # track model and experiment configurations
        config={
            "exp_config": exp_config,
            "model_config": sat_config,
        }
    )

    wandb.define_metric("acc", summary="max")

    run_dual_encoders(wandb.config)

    return

run_dual_encoders(config = {"model_config": sat_config, "exp_config": exp_config})

# sweep_configuration = {
#     "method": "bayes",
#     "metric": {'goal': 'maximize', 'name': 'acc'},
#     "parameters": {
#         "model_config" : {
#             "parameters": {
#                 "model_type": {"values":["graph_benchmarks"]},
#                 "vocab_size": {"values":[len(tokens)]},
#                 "embedding_dim": {"values":[128]},
#                 "in_embed": {"values":[False]},
#                 "abs_pe": {"values":[True, False]},
#                 "abs_pe_dim": {"values":[128]},
#                 "use_edge_attr": {"values":[True, False]},
#                 "dim_feedforward": {"values": [256]},
#                 "num_heads": {"values": [8]},
#                 "num_layers": {"values": [4]},
#                 "se": {"values": ['pna']},
#                 "dropout": {"values": [0.2]},
#                 "gnn_layers": {"values": [0,4]},
#                 "directed_attention": {"values": [True,False]}
#             }
#         }
#     }
# }
#
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='hol4_premise_selection')
# #
# wandb.agent(sweep_id,function=main)
#
#
#






































#todo add test set evaluation

##########################################
##########################################

#meta graph

##########################################
##########################################

# num_nodes = len(enc_nodes.get_feature_names_out())
#
# def new_batch_loss(batch, struct_net_target,
#                    struct_net_source,
#                    graph_net, theta1, theta2, theta3,
#                    gamma1, gamma2, b1, b2):
#     B = len(batch)
#
#     def phi_1(x):
#         return inner_embedding_network.phi(x, gamma1, b1)
#
#     def phi_2(x):
#         return inner_embedding_network.phi(x, gamma2, b2)
#
#     x_t_struct = struct_net_target(inner_embedding_network.sp_to_torch(sp.sparse.vstack(batch.x_t_one_hot)).to(device))
#     x_s_struct = struct_net_source(inner_embedding_network.sp_to_torch(sp.sparse.vstack(batch.x_s_one_hot)).to(device))
#
#     x_t_attr = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(
#     device))
#
#     x_s_attr = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))
#
#
#     # x_t_attr = x_t_attr.reshape(x_t_attr.shape[1],x_t_attr.shape[0])
#     # x_s_attr = x_s_attr.reshape(x_s_attr.shape[1],x_s_attr.shape[0])
#
#     sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     attr_sim = sim_func(x_t_attr, x_s_attr)
#     struct_sim = sim_func(x_t_struct, x_s_struct)
#     align_sim = sim_func(x_t_attr, x_s_struct)
#
#     # print ((1/32) * sum(batch.y * struct_sim + ((1. - batch.y) * -1. * attr_sim)))
#
#     attr_loss = inner_embedding_network.loss_similarity(attr_sim, batch.y.to(device), phi_1, phi_2)
#     struct_loss = inner_embedding_network.loss_similarity(struct_sim, batch.y.to(device), phi_1, phi_2)
#     align_loss = inner_embedding_network.loss_similarity(align_sim, batch.y.to(device), phi_1, phi_2)
#
#     tot_loss = theta1 * attr_loss + theta2 * struct_loss + theta3 * align_loss
#     return (1. / B) * torch.sum(tot_loss)
#
#
#
#
#
# def run(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)
#
#     optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     training_losses = []
#
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#
#             optimiser_fc.zero_grad()
#
#             optimiser_gnn.zero_grad()
#
#             loss_val = loss(graph_net, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#             loss_val.backward()
#
#             optimiser_fc.step()
#
#             optimiser_gnn.step()
#
#             training_losses.append(loss_val.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 #todo: val moving average, every e.g. 25 record val, then take avg at 1000
#
#                 validation_loss = accuracy(graph_net, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#
#                 val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#         #print ("Epoch {} done".format(j))
#
#     return training_losses, val_losses
#
#
#
# def run_meta(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations):
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     struct_net_source = inner_embedding_network.StructureEmbeddingSource(num_nodes, embedding_dim * 2).to(device)
#
#     struct_net_target = inner_embedding_network.StructureEmbeddingTarget(num_nodes, embedding_dim * 2).to(device)
#
#     optimiser_struct_source = torch.optim.Adam(list(struct_net_source.parameters()), lr=step_size,
#                                                weight_decay=decay_rate)
#
#     optimiser_struct_target = torch.optim.Adam(list(struct_net_target.parameters()), lr=step_size,
#                                                weight_decay=decay_rate)
#
#     training_losses = []
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#             optimiser_gnn.zero_grad()
#             optimiser_struct_target.zero_grad()
#             optimiser_struct_source.zero_grad()
#
#             theta1 = 0.7
#             theta2 = 0.2
#             theta3 = 0.1
#
#
#             loss_val = new_batch_loss(batch, struct_net_target,
#                                       struct_net_source,
#                                       graph_net, theta1, theta2, theta3,
#                                       gamma1=2, gamma2=2, b1=0.1, b2=0.1)
#
#             loss_val.backward()
#
#             optimiser_gnn.step()
#             optimiser_struct_target.step()
#             optimiser_struct_source.step()
#
#             training_losses.append(loss_val.detach() / batch_size)
#             # print (loss_val.detach())
#
#     return training_losses#, val_losses


# def val_batch(batch, struct_net_target,
#               struct_net_source,
#               graph_net, theta1, theta2, theta3,
#               gamma1, gamma2, b1, b2):
#
#     x_t_struct = struct_net_target(sp_to_torch(sp.sparse.vstack(batch.x_t_one_hot)).to(device))
#     x_s_struct = struct_net_source(sp_to_torch(sp.sparse.vstack(batch.x_s_one_hot)).to(device))
#
#     x_t_attr = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device)).to(device)
#
#     x_s_attr = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), F_p, F_i, F_o, F_x, conv1, conv2,
#                          num_iterations, batch.x_s_batch.to(device)).to(device)
#
#     sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     attr_sim = sim_func(x_t_attr, x_s_attr)
#     struct_sim = sim_func(x_t_struct, x_s_struct)
#     align_sim = sim_func(x_t_attr, x_s_struct)
#
#     scores = 0.5 * attr_sim + 0.5 * align_sim
#
#     preds = (scores > 0).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(
#         batch.y)  # best_score(list(zip(scores, batch.y.to(device))))#curr_best_acc, curr_best_lam
#
#
#




    # find scoring threshold which gives the best prediction metric (accuracy for now)
    #     def best_score(scores):
    #         #sort scores
    #         sorted_scores = sorted(scores, key=lambda tup: tup[0])

    #         #print (sorted_scores[:10])

    #         #evaluate metric (accuracy here) using sorted scores and index
    #         def metric_given_threshold(index):
    #             pos = scores[index:]
    #             neg = scores[:index]

    #             correct = len([x for x in pos if x[1] == 1]) + len([x for x in neg if x[1] == 0])

    #             return correct / len(sorted_scores)

    #         #loop through indices testing best threshold
    #         curr_best_metric = 0.
    #         curr_best_idx = 0

    #         for i in range(len(sorted_scores)):
    #             new = metric_given_threshold(i)
    #             if new > curr_best_metric:
    #                 curr_best_metric = new
    #                 curr_best_idx = i

    #         return curr_best_metric, curr_best_idx

    #     #only need one lambda when doing inductive since there's only 2 values to weigh
    #     lam_grid = np.logspace(-1,1,10)

    #     #grid search over lambda for best score

    #     curr_best_lam = 0
    #     curr_best_acc = 0

    #     for lam in lam_grid:
    #         scores = []
    #         for (x1,x2,y) in sims:
    #             scores.append((x1 + lam * x2, y))

    #         acc, idx = best_score(scores)
    #         if acc > curr_best_acc:
    #             curr_best_acc = acc
    #             curr_best_lam = lam

    # keep lambda as thetas for now











