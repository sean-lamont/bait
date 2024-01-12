import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn.functional import dropout
import torch.nn.functional as F


class CombinedAggregation(nn.Module):
    def __init__(self, embedding_dim, batch_norm=True):
        super(CombinedAggregation, self).__init__()
        # self.fc = nn.Linear(embedding_dim, embedding_dim)
        # self.bn = nn.BatchNorm1d(embedding_dim)

        if batch_norm:
            self.mlp = torch.nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU()
            )
        else:
            self.mlp = torch.nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.mlp(x)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, input_shape):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape)
        self.fc2 = nn.Linear(input_shape, 1)
        self.bn = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))


#####################################################################################################
# FormulaNet with no edge attributes
#####################################################################################################

# F_o summed over children
class ChildAggregation(MessagePassing):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__(aggr='sum', flow='target_to_source')
        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())
        else:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def forward(self, x, edge_index):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)


# F_i summed over parents
class ParentAggregation(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__(aggr='sum', flow='source_to_target')
        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())
        else:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)

        return self.mlp(tmp)

    def forward(self, x, edge_index):
        # edge index 1 for degree wrt parents
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0

        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)



class FormulaNet(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, batch_norm=True):
        super(FormulaNet, self).__init__()

        self.num_iterations = num_iterations

        self.initial_encoder = nn.Embedding(input_shape, embedding_dim)
        self.parent_agg = ParentAggregation(embedding_dim, embedding_dim, batch_norm=batch_norm)
        self.child_agg = ChildAggregation(embedding_dim, embedding_dim, batch_norm=batch_norm)
        self.final_agg = CombinedAggregation(embedding_dim, batch_norm=batch_norm)

    def forward(self, data):
        nodes = data.x
        edges = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # nodes = self.initial_encoder(nodes)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)
            fo_sum = self.child_agg(nodes, edges)
            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update

        return gmp(nodes, batch)



#####################################################################################################
# FormulaNet with edge attributes
#####################################################################################################


class ChildAggregationEdges(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=32, batch_norm = True):
        super().__init__(aggr='sum', flow='target_to_source')

        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels + edge_dim, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())

        else:
            self.mlp = Seq(Linear(2 * in_channels + edge_dim, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)

class ParentAggregationEdges(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=32, batch_norm = True):
        super().__init__(aggr='sum', flow='source_to_target')


        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels + edge_dim, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())

        else:
            self.mlp = Seq(Linear(2 * in_channels + edge_dim, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())


    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


class RegressionMLP(nn.Module):
    def __init__(self, input_shape):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape)
        self.fc2 = nn.Linear(input_shape, 1)
        self.bn = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class FormulaNetEdges(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, max_edges=200,
                 edge_dim = 32, in_encoder=None, global_pool=True, batch_norm=True):
        super(FormulaNetEdges, self).__init__()
        self.num_iterations = num_iterations
        self.global_pool = global_pool

        if in_encoder is not None:
            self.initial_encoder = in_encoder
        else:
            self.initial_encoder = nn.Embedding(input_shape, embedding_dim)

        # assume max 200 children
        self.edge_encoder = nn.Embedding(max_edges, edge_dim)
        self.parent_agg = ParentAggregationEdges(embedding_dim, embedding_dim, batch_norm=batch_norm)#,edge_dim=64)
        self.child_agg = ChildAggregationEdges(embedding_dim, embedding_dim, batch_norm=batch_norm)#, edge_dim=64)
        self.final_agg = CombinedAggregation(embedding_dim, batch_norm=batch_norm)

    def forward(self, batch):  # nodes, edges, edge_attr, batch=None):
        nodes = batch.x
        edges = batch.edge_index
        edge_attr = batch.edge_attr

        if hasattr(batch, 'node_depth'):
            nodes = self.initial_encoder(nodes, batch.node_depth.view(-1))
        else:
            nodes = self.initial_encoder(nodes)
        edge_attr = self.edge_encoder(edge_attr)  # torch.transpose(edge_attr, 0, 1))

        if len(edge_attr.shape) > 2:
            edge_attr = edge_attr.flatten(-2, -1)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges, edge_attr)
            fo_sum = self.child_agg(nodes, edges, edge_attr)
            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update


        if self.global_pool:
            return gmp(nodes, batch.batch)
        else:
            return nodes


#####################################################################################################
# GNN for induction (term) network
#####################################################################################################

class Final_Agg_induct(nn.Module):
    def __init__(self, embedding_dim):
        super(Final_Agg_induct, self).__init__()
        self.fc = nn.Linear(embedding_dim * 3, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, x):
        x = self.fc2(dropout(torch.relu(self.fc(dropout(x)))))
        return x



class message_passing_gnn_induct(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, device):
        super(message_passing_gnn_induct, self).__init__()

        self.device = device

        self.num_iterations = num_iterations

        # self.initial_encoder = torch.nn.Linear(input_shape, embedding_dim).to(device)
        self.initial_encoder = torch.nn.Embedding(input_shape, embedding_dim)

        self.parent_agg = ParentAggregation(embedding_dim, embedding_dim)

        self.child_agg = ChildAggregation(embedding_dim, embedding_dim)

        self.final_agg = Final_Agg_induct(embedding_dim)

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim * 2, 1, stride=1)

    def forward(self, nodes, edges, batch=None):
        nodes = self.initial_encoder(nodes)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)

            fo_sum = self.child_agg(nodes, edges)

            node_update = self.final_agg(torch.cat([nodes, fi_sum, fo_sum], axis=1).to(self.device))

            nodes = nodes + node_update

        nodes = nodes.unsqueeze(-1)

        nodes = self.conv1(nodes)

        nodes = nodes.squeeze(-1)

        #        g_embedding = torch.cat([gmp(nodes, batch), gap(nodes, batch)], dim=1)  # gmp(nodes, batch)
        # return embeddings for each node which is a variable
        return nodes

class FormulaNetSAT(nn.Module):
    def __init__(self, embedding_dim, num_iterations, batch_norm=True, edge_dim=32):
        super(FormulaNetSAT, self).__init__()
        self.num_iterations = num_iterations


        # assume max 200 children
        self.parent_agg = ParentAggregationEdges(embedding_dim, embedding_dim, batch_norm=batch_norm,edge_dim=edge_dim)#,edge_dim=64)
        self.child_agg = ChildAggregationEdges(embedding_dim, embedding_dim, batch_norm=batch_norm,edge_dim=edge_dim)#, edge_dim=64)
        self.final_agg = CombinedAggregation(embedding_dim, batch_norm=batch_norm)

    def forward(self, nodes, edges, edge_attr):

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges, edge_attr)
            fo_sum = self.child_agg(nodes, edges, edge_attr)
            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update

        return nodes