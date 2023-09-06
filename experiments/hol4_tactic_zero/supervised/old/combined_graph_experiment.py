import traceback
import wandb
import random
import models.gnn_edge_labels
from models.graph_transformers.SAT.sat.models import GraphTransformer
import math
import torch
import torch_geometric.utils
from tqdm import tqdm
from models.gnn.digae.digae_model import OneHotDirectedGAE
import json
import models.gnn.formula_net.inner_embedding_network
from torch_geometric.data import Data
import pickle
from data.hol4.ast_def import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"


with open("/data/hol4/old/dep_data.json") as fp:
    dep_db = json.load(fp)
    
with open("/data/hol4/old/new_db.json") as fp:
    new_db = json.load(fp)

#with open("polished_dict.json") as f:
#    p_d = json.load(f)


with open("/data/hol4/old/torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

# with open("one_hot_dict.pk", "rb") as f:
#     one_hot_dict = pickle.load(f)

with open("/data/hol4/old/train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)

polished_goals = []
for val_ in new_db.values():
    polished_goals.append(val_[2])


# todo load this from somewhere!
tokens = list(set([token.value for polished_goal in polished_goals for token in polished_to_tokens_2(polished_goal)  if token.value[0] != 'V']))

tokens.append("VAR")
tokens.append("VARFUNC")
tokens.append("UNKNOWN")



# Data class for combining two graphs into one for supervised learning (i.e. premise selection). num_nodes_g1 tells us where in data.x to index to get nodes from g1
class CombinedGraphData(Data):
    def __init__(self, combined_x, combined_edge_index, num_nodes_g1, y, combined_edge_attr=None, complete_edge_index=None, pos_enc=None):
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

        self.pos_enc = pos_enc


def get_directed_edge_index(num_nodes, edge_idx):
    from_idx = []
    to_idx = []

    for i in range(0,num_nodes-1):
        # to_idx = [i]
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx)
        except:
            print (f"exception {i, num_nodes, edge_idx}")

        # ancestor_nodes = ancestor_nodes.item()
        found_nodes = list(ancestor_nodes).remove(i)

        if found_nodes is not none:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx, flow='target_to_source')
        # children_nodes = children_nodes.item()
        # print (found_nodes, children_nodes, i, self_idx.item(), edge_idx)
        found_nodes = list(children_nodes).remove(i)

        if found_nodes is not none:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)

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


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

def positional_encoding(d_model, depth_vec):
    size, _ = depth_vec.shape

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(size, d_model)

    pe[:, 0::2] = torch.sin(depth_vec * div_term)
    pe[:, 1::2] = torch.cos(depth_vec * div_term)

    return pe


# todo port to MongoDB
def to_combined_batch(list_data, data_dict, embedding_dim, directed_attention=False):
    batch_list = []
    for (x1, x2, y) in list_data:
        # x1/x_t is conj, x2/x_s is stmt
        conj = x1
        stmt = x2

        conj_graph = data_dict[conj]
        stmt_graph = data_dict[stmt]

        # batch_list.append(
        #     LinkData(edge_index_s=stmt_graph.edge_index, x_s=stmt_graph.x, edge_attr_s=stmt_graph.edge_attr, edge_index_t=conj_graph.edge_index, x_t=conj_graph.x, edge_attr_t=conj_graph.edge_attr, y=torch.tensor(y)))

        # Concatenate node feature matrices
        combined_features = torch.cat([conj_graph.x, stmt_graph.x], dim=0)

        # Combine edge indices
        num_nodes_g1 = conj_graph.num_nodes
        edge_index1 = conj_graph.edge_index
        edge_index2 = stmt_graph.edge_index +num_nodes_g1
        combined_edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        # combine edge attributes
        edge_attr1 = conj_graph.edge_attr
        edge_attr2 = stmt_graph.edge_attr
        combined_edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)


        #directed edge index case
        if directed_attention:
            complete_edge_index1 = conj_graph.complete_edge_index
            complete_edge_index2 = stmt_graph.complete_edge_index + num_nodes_g1

            complete_edge_index = torch.cat([complete_edge_index1, complete_edge_index2], dim =1)

        else:
        # Compute disjoint pairwise complete edge indices
            complete_edge_index1 = torch.cartesian_prod(torch.arange(num_nodes_g1),
                                                        torch.arange(num_nodes_g1))  # All pairs of nodes in conj_graph


            complete_edge_index2 = torch.cartesian_prod(torch.arange(num_nodes_g1, num_nodes_g1 + stmt_graph.num_nodes),
                                                        torch.arange(num_nodes_g1,
                                                                    num_nodes_g1 + stmt_graph.num_nodes))  # All pairs of nodes in stmt_graph

            complete_edge_index = torch.cat([complete_edge_index1, complete_edge_index2], dim=0).t().contiguous()


        # positional encodings

        graph_ind = torch.cat([torch.ones(num_nodes_g1), torch.ones(stmt_graph.num_nodes) * 2], dim=0)
        pos_enc = positional_encoding(embedding_dim, graph_ind.unsqueeze(1))

        #append combined graph to batch

        batch_list.append(CombinedGraphData(combined_x=combined_features,combined_edge_index=combined_edge_index, combined_edge_attr=combined_edge_attr, complete_edge_index=complete_edge_index, num_nodes_g1=num_nodes_g1, pos_enc=pos_enc, y=y))

    loader = iter(DataLoader(batch_list, batch_size=len(batch_list)))

    batch = next(iter(loader))

    return batch



def get_model(config):

    if config['model_type'] == 'graph_benchmarks':
        return GraphTransformer(in_size=config['vocab_size'], num_class=2,
                                d_model=config['embedding_dim'],
                                dim_feedforward=config['dim_feedforward'],
                                num_heads=config['num_heads'],
                                num_layers=config['num_layers'],
                                in_embed=config['in_embed'],
                                se=config['se'],
                                abs_pe=config['abs_pe'],
                                abs_pe_dim=config['abs_pe_dim'],
                                use_edge_attr=config['use_edge_attr'],
                                dropout=config['dropout'],
                                # global_pool=config['global_pool'],
                                global_pool='cls',
                                k_hop=config['gnn_layers'],
                                num_edge_features=200)

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


# def run_dual_encoders(model_config, exp_config):
def run_combined_graphs(config):

    model_config = config['model_config']
    exp_config = config['exp_config']

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    graph_net_combined = get_model(model_config).to(device)

    # graph_net_combined = torch.nn.DataParallel(graph_net_combined)#.to(device)
    # graph_net_combined.cuda()

    print ("Model details:")

    print(graph_net_combined)
    #
    wandb.log({"Num_model_params": sum([p.numel() for p in graph_net_combined.parameters() if p.requires_grad])})

    embedding_dim = model_config['embedding_dim']
    lr = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    save = exp_config['model_save']
    val_size = exp_config['val_size']
    if 'directed_attention' in model_config:
        directed_attention = model_config['directed_attention']
    else:
        directed_attention = False

    #wandb load
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']

    print (torch.cuda.device_count())
    fc = models.gnn_edge_labels.BinaryClassifier(embedding_dim).to(device)
    # fc = torch.nn.DataParallel(fc)#.to(device)
    # fc = fc.cuda()

    op_g = torch.optim.AdamW(graph_net_combined.parameters(), lr=lr, weight_decay=weight_decay)
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
                batch = to_combined_batch(train_data[from_idx:to_idx], torch_graph_dict, embedding_dim, directed_attention = directed_attention)
            except Exception as e:
                # print(f"Error in batch {i}: {e}")
                continue

            # op_enc.zero_grad()
            op_g.zero_grad()
            op_fc.zero_grad()


            # positional encoding for which graph is first/second :

            # data = Data(x=batch.x.float().cuda(), edge_index=batch.edge_index.cuda(),
            #             batch=batch.batch.cuda(),
            #             ptr=batch.ptr.cuda(),
            #             complete_edge_index=batch.complete_edge_index.cuda(),
            #             edge_attr=batch.combined_edge_attr.long().cuda(),
            #             abs_pe=batch.pos_enc.cuda())#,

            data = Data(x=batch.x.float().to(device), edge_index=batch.edge_index.to(device),
                          batch=batch.batch.to(device),
                          ptr=batch.ptr.to(device),
                          complete_edge_index=batch.complete_edge_index.to(device),
                          edge_attr=batch.combined_edge_attr.long().to(device),
                          abs_pe=batch.pos_enc.to(device))#,


            try:
                combined_graph_enc = graph_net_combined(data)

                preds = fc(combined_graph_enc)

                eps = 1e-6

                preds = torch.clip(preds, eps, 1 - eps)

                # loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))
                loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))


                loss.backward()

                # op_enc.step()
                op_g.step()
                op_fc.step()


            except Exception as e:
                err_count += 1
                if err_count > 100:
                    return Exception("Too many errors in training")
                print(f"Error in training {e}")
                print (traceback.format_exc())
                # print (data.edge_attr)
                continue

            training_losses.append(loss.detach() / batch_size)

            if i % (10000 // batch_size) == 0:

                graph_net_combined.eval()

                val_count = []

                random.shuffle(val_data)

                val_inds = list(range(0, len(val_data), batch_size))
                val_inds.append(len(val_data) - 1)


                for k in tqdm(range(0, val_size // batch_size)):
                    val_err_count = 0

                    from_idx_val = val_inds[k]
                    to_idx_val = val_inds[k + 1]

                    try:
                        val_batch = to_combined_batch(val_data[from_idx_val:to_idx_val], torch_graph_dict, embedding_dim)
                        validation_loss = val_acc_combined(graph_net_combined, val_batch, fc, embedding_dim)
                        val_count.append(validation_loss.detach())
                    except Exception as e:
                        # print(f"Error {e}, batch: {val_batch}")
                        # print (e)
                        print (traceback.print_exc())
                        val_err_count += 1
                        continue

                print (f"Val errors: {val_err_count}")

                validation_loss = (sum(val_count) / len(val_count)).detach()
                val_losses.append((validation_loss, j, i))

                print("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print("Val acc: {}".format(validation_loss.detach()))

                print (f"Failed batches: {err_count}")

                wandb.log({"acc": validation_loss.detach(),
                           "train_loss_avg": sum(training_losses[-100:]) / len(training_losses[-100:]),
                           "epoch": j})

                if validation_loss > best_acc:
                    best_acc = validation_loss
                    print(f"New best validation accuracy: {best_acc}")
                    # only save encoder if best accuracy so far
                    if save == True:
                        torch.save(graph_net_combined, "combined_graph_gnn")

                    # w)andb save
                    # torch.save({  # Save our checkpoint loc
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss,
                    # }, CHECKPOINT_PATH)
                    # wandb.save(CHECKPOINT_PATH)  # saves c


                graph_net_combined.train()

        wandb.log({"failed_batches": err_count})
    print(f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses


def val_acc_combined(model,  batch, fc, embedding_dim):
    # data = Data(x=batch.x.float().cuda(), edge_index=batch.edge_index.cuda(),
    #             batch=batch.batch.cuda(),
    #             ptr=batch.ptr.cuda(),
    #             complete_edge_index=batch.complete_edge_index.cuda(),
    #             edge_attr=batch.combined_edge_attr.long().cuda(),
    #             abs_pe=batch.pos_enc.cuda())  # ,

    data = Data(x=batch.x.float().to(device), edge_index=batch.edge_index.to(device),
                batch=batch.batch.to(device),
                ptr=batch.ptr.to(device),
                complete_edge_index=batch.complete_edge_index.to(device),
                edge_attr=batch.combined_edge_attr.long().to(device),
                abs_pe=batch.pos_enc.to(device))#,


    graph_enc = model(data)

    preds = fc(graph_enc)

    preds = torch.flatten(preds)

    preds = (preds > 0.5).long()

    # return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)
    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


# data_config = {

#
# }


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
    "abs_pe_dim": 128,
    "use_edge_attr": True,
    "dropout": 0.2,
    "gnn_layers": 4,
    "global_pool": "cls",
    "directed_attention": True
}

exp_config = {
    "learning_rate": 1e-4,
    "epochs": 8,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "model_save": False,
    "val_size": 2048,
}

formula_net_config = {
    "model_type": "formula-net",
    "vocab_size": len(tokens),
    "embedding_dim": 256,
    "gnn_layers": 4,
}

#initialise with default parameters

# run_combined_graphs(config={"model_config": sat_config, "exp_config":exp_config})


# training_losses, val_losses = run_combined_graphs(
# config={
#     "exp_config": exp_config,
#     "model_config": sat_config,
# })

with open("/home/sean/Documents/phd/aitp/experiments/hol4/supervised/torch_graph_dict_directed_depth.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

def main():
    wandb.init(
        project="hol4_premise_selection",

        name="Directed + Combined Graph graph_benchmarks",
        # track model and experiment configurations
        config={
            "exp_config": exp_config,
            "model_config": sat_config,
        }
    )

    wandb.define_metric("acc", summary="max")

    run_combined_graphs(wandb.config)

    return

# main()

sweep_configuration = {
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'acc'},
    "parameters": {
        "model_config" : {
            "parameters": {
                "model_type": {"values":["graph_benchmarks"]},
                "vocab_size": {"values":[len(tokens)]},
                "embedding_dim": {"values":[128]},
                "in_embed": {"values":[False]},
                "abs_pe": {"values":[True, False]},
                "abs_pe_dim": {"values":[128]},
                "use_edge_attr": {"values":[True, False]},
                "dim_feedforward": {"values": [512]},
                "num_heads": {"values": [8]},
                "num_layers": {"values": [4]},
                "se": {"values": ['pna']},
                "dropout": {"values": [0.2]},
                "gnn_layers": {"values": [0, 4]},
                "directed_attention": {"values": [True,False]}
            }
        }
    }
}
#

# sweep_configuration = {
#     "method": "bayes",
#     "metric": {'goal': 'maximize', 'name': 'acc'},
#     "parameters": {
#         "model_config" : {
#             "parameters": {
#                 "model_type": {"values":["formula-net"]},
#                 "vocab_size": {"values":[len(tokens)]},
#                 "embedding_dim": {"values":[256]},
#                 "gnn_layers": {"values": [0, 1, 2, 4]},
#             }
#         }
#     }
# }
sweep_id = wandb.sweep(sweep=sweep_configuration, project='hol4_premise_selection')
#
wandb.agent(sweep_id,function=main)


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
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))#
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











