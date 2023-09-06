import traceback
from environments.hol4 import graph_env
from datetime import datetime
from data.hol4 import ast_def
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.tactic_zero import policy_models
from models.gnn.formula_net import inner_embedding_network
import time
from environments.hol4.graph_env import *
import numpy as np

#import batch_gnn




MORE_TACTICS = True
if not MORE_TACTICS:
    thms_tactic = ["simp", "fs", "metis_tac"]
    thm_tactic = ["irule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac"]
else:
    thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    thm_tactic = ["irule", "drule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac", "EQ_TAC"]
    
tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic

#TODO Move to another file 
#

# def get_polish(raw_goal):
#         goal = construct_goal(raw_goal)
#         process.sendline(goal.encode("utf-8"))
#         process.expect("\r\n>")
#         process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
#         process.expect("\r\n>")
#         process.sendline("top_goals();".encode("utf-8"))
#         process.expect("val it =")
#         process.expect([": goal list", ":\r\n +goal list"])
#
#         polished_raw = process.before.decode("utf-8")
#         polished_subgoals = re.sub("“|”","\"", polished_raw)
#         polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)
#
#         pd = eval(polished_subgoals)
#
#         process.expect("\r\n>")
#         process.sendline("drop();".encode("utf-8"))
#         process.expect("\r\n>")
#         process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
#         process.expect("\r\n>")
#
#         data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
#                  "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
#                 for e in zip(pd, [([], raw_goal)])]
#         return data
#
# def construct_goal(goal):
#     s = "g " + "`" + goal + "`;"
#     return s
#
#
# def parse_theory(pg):
#     theories = re.findall(r'C\$(\w+)\$ ', pg)
#     theories = set(theories)
#     for th in EXCLUDED_THEORIES:
#         theories.discard(th)
#     return list(theories)

def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions): 
        #goal = "@ @ D$min$==> {} {}".format(i, goal)
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)

    return goal 

def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs


'''

High level agent class 

'''
class Agent:
    def __init__(self, tactic_pool):
        self.tactic_pool = tactic_pool    
        self.load_encoder()
    
    def load_agent(self):
        pass
    
    def load_encoder(self):
        pass
        
    def run(self, env, max_steps):
        pass
    
    def update_params(self):
        pass
    
    
    



#gnn_enc = torch.load("model_checkpoints/gnn_encoder_latest")#.eval()
#
# for p in gnn_enc.parameters():
#     p.require_grads = False


with open("../../../data/hol4/old/torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def nodes_list_to_senders_receivers(node_list):
    senders = []
    receivers = []
    for i, node in enumerate(node_list):
        for child in node.children:
            senders.append(i)
            receivers.append(node_list.index(child))
    return senders, receivers

def nodes_list_to_senders_receivers_labelled(node_list):
    senders = []
    receivers = []
    edge_labels = []
    for i, node in enumerate(node_list):
        for j, child in enumerate(node.children):
            senders.append(i)
            receivers.append(node_list.index(child))
            edge_labels.append(j)
    return senders, receivers, edge_labels


# # Return set of all unique nodes in a graph
# def nodes_list(g, result=[]):
#     result.append(g)
#
#     for val in g.children.values():
#         siblings = val[1]
#         for sibling in siblings:
#             nodes_list(sibling, result)
#
#     return list(set(result))


def nodes_list(g, result=[]):
    result.append(g)
    for child in g.children:
        nodes_list(child, result)

    return list(set(result))


with open("../../../data/hol4/old/graph_token_encoder.pk", "rb") as f:
    token_enc = pickle.load(f)

def sp_to_torch(sparse):
    coo = sparse.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))  # .to_dense()


def graph_to_torch(g):
    node_list = nodes_list(g, result=[])
    senders, receivers = nodes_list_to_senders_receivers(node_list)

    # get the one hot encoding from enc
    t_f = lambda x: np.array([x.node.value])

    node_features = list(map(t_f, node_list))

    node_features = token_enc.transform(node_features)

    edges = torch.tensor([senders, receivers], dtype=torch.long)

    nodes = sp_to_torch(node_features)

    return Data(x=nodes, edge_index=edges)

def graph_to_torch_labelled(g):
    node_list = nodes_list(g, result=[])
#    senders, receivers = nodes_list_to_senders_receivers(node_list)
    senders, receivers, edge_labels = nodes_list_to_senders_receivers_labelled(node_list)



    # define labels before renaming to keep original variables for induction
    labels = [x.node.value for x in node_list]

    # rename variables to be constant
    for node in node_list:
        if node.node.value[0] == 'V':
            if node.children != []:
                node.node.value = "VARFUNC"
            else:
                node.node.value = "VAR"

    # get the one hot encoding from enc
    t_f = lambda x: np.array([x.node.value])

    node_features = list(map(t_f, node_list))

    node_features = token_enc.transform(node_features)

    edges = torch.tensor([senders, receivers], dtype=torch.long)

    nodes = sp_to_torch(node_features)

    return Data(x=nodes, edge_index=edges, edge_attr=torch.Tensor(edge_labels), labels=labels)

# #make database compatible with GNN encoder
encoded_graph_db = []
with open('../../../../data/hol4/data_v2/data/include_probability.json') as f:
    compat_db = json.load(f)
    
reverse_database = {(value[0], value[1]) : key for key, value in compat_db.items()}


graph_db = {}

print ("Generating premise graph db...")
for i,t in enumerate(compat_db):

#    graph_db[t] = graph_to_torch(ast_def.goal_to_graph(t))
    graph_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t))

with open("../../../data/hol4/old/paper_goals.pk", "rb") as f:
   paper_goals = pickle.load(f)

# only take valid goals with database
# valid_goals
# for goal in paper_goals:
#    if goal[0] in compat_db.keys():
#        valid_goals.append(goal)
#
# print (f"Len valid {len(valid_goals)}")
# np.random.shuffle(valid_goals)
#
# with open("valid_goals_shuffled.pk", "wb") as f:
#    pickle.dump(valid_goals, f)
#


with open("../../../data/hol4/old/valid_goals_shuffled.pk", "rb") as f:
    valid_goals = pickle.load(f)

train_goals = valid_goals[:int(0.8 * len(valid_goals))]
test_goals = valid_goals[int(0.8 * len(valid_goals)):]

#compat_goals = paper_goals
# def gather_encoded_content_gnn_new(graph, encoder):
#     fringe_sizes = []
#     contexts = []
#     reverted = []
#     enc_dict = {}
#
#     # for i in history:
#     #     c = i["content"]
#     #     contexts.extend(c)
#     #     fringe_sizes.append(len(c))
#     # for e in contexts:
#     #     g = revert_with_polish(e)
#     #     reverted.append(g)
#     #
#
#     # print ("encoding...")
#     # print (len(graph))
#
#     # if len(graph) > 1:
#     #     print ([g.goal for g in graph])
#     # graph for now just list of goals
#     for goal in graph:
#         g = revert_with_polish(goal.goal)
#         reverted.append(g)
#
#     # print ("a")
#
# #    graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch(ast_def.goal_to_graph(t)) for t in reverted]
#     graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t)) for t in reverted]
#
#     # print ("b")
#     loader = DataLoader(graphs, batch_size = len(reverted))
#
#     batch = next(iter(loader))
#
#     # representations = torch.unsqueeze(encoder.forward(batch.x.to(device), batch.edge_indegraph_pow_defdevice), batch.batch.to(device)), 1)
#     #encode_and_pool for digae model
#     # todo temporary cache of representations to avoid recomputing encoding for same goal in same proof attempt (may not be worth it since batched call is fast anyway
#
#
#     representations = torch.unsqueeze(encoder.encode_and_pool(batch.x.to(device), batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)), 1)
#
#
#     print (representations.shape)
#
#     for i in range(len(graph)):
#         graph[i].embedding = representations[i]
#
#     # print ("c")
#     return representations

# updown = UpDown(agg_siblings=AggSiblings(device), agg_children=AggChildren(), agg_context=AggContext(device), device=device)

def get_best_goal(goals):
    # take product of context and goal scores for now
    for goal in goals:
        goal.final_score = goal.agg_score * goal.context_score

    goal_scores = [goal.final_score for goal in goals]
    goal_probs = F.softmax(goal_scores, dim=0)

    goal_m = Categorical(goal_probs)
    goal = goal_m.sample()
    # fringe_pool.append(goal_m.log_prob(goal))

    return goals[goal]

def select_best_subgoal(node):
    # take raw_score since we have already chosen path to take, don't need context
    # raw score of node vs agg score of children tells us which down path is most promising
    best_score = node.raw_score
    best_node = node
    flag = False
    for tac, (tac_score, siblings) in node.children.items():
        # using tac_score since it is prob of proving goal using that tactic (ignoring context since we've already chosen parent path)
        if tac_score > best_score:
            best_score = tac_score
            # doesn't matter which sibling is attacked first since all need to be proven
            best_node = siblings[0]
            # update flag signalling to run again with new node
            flag = True

    if flag:
        return select_best_subgoal(best_node)
    else:
        return best_node

def embed_goal_graph_replay(main_goal, graph, encoder, score_net, chosen_goal, updown):
    reverted = []

    for goal in graph:

        g = revert_with_polish(goal.goal)
        reverted.append(g)

    graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t)) for t in reverted]

    loader = DataLoader(graphs, batch_size = len(reverted))

    batch = next(iter(loader))

    representations = torch.unsqueeze(encoder.encode_and_pool(batch.x.to(device), batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)), 1)


    initial_vecs = score_net(representations).squeeze(-1).squeeze(-1)

    for i in range(len(graph)):
        graph[i].initial_vec = initial_vecs[i]
    # run up-down algorithm on graph

    # todo not sure if this correctly generates autograd graph?
    updown.up_step(main_goal)
    updown.down_step(main_goal)

    # take product of context and goal scores for now

    for goal in graph:
        goal.final_score = updown.final_agg(goal)

    goal_scores = [goal.final_score.squeeze(0) for goal in graph]
    goal_scores = torch.cat(goal_scores, dim=0)

    goal_probs = F.softmax(goal_scores, dim=0)
        # print (f"goal prob shape {goal_probs.shape}")
        # print (f"goal probs {goal_probs}")
        # goal_probs = torch.FloatTensor(goal_scores)


    goal_m = Categorical(goal_probs)

    chosen_idx = graph.index(chosen_goal)
    # chosen_prob = goal_m.log_prob(chosen_idx)

    return representations[chosen_idx], goal_m.log_prob(torch.tensor(chosen_idx).to(device))


def embed_goal_graph(main_goal, graph, encoder, score_net, updown):
    reverted = []

    for goal in graph:
        g = revert_with_polish(goal.goal)
        reverted.append(g)

    graphs = [graph_db[t] if t in graph_db.keys() else graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t)) for t in reverted]

    loader = DataLoader(graphs, batch_size = len(reverted))

    batch = next(iter(loader))

    representations = torch.unsqueeze(encoder.encode_and_pool(batch.x.to(device), batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)), 1)



    initial_vecs = score_net(representations).squeeze(-1).squeeze(-1)

    # print ("initial shape \n")
    # print (initial_vecs.shape)
    try:

        for i in range(len(graph)):
            graph[i].initial_vec = initial_vecs[i]

        # run up-down algorithm on graph

        # if len(graph) > 3:
        #     print ("goal graph before\n")
        #     main_goal._print_with_scores()


        # print (f"graph vec shape {graph[0].initial_vec.shape}")
        updown.up_step(main_goal)

        # if len(graph) > 3:
        #     print ("goal graph after up\n")
        #     main_goal._print_with_scores()

        updown.down_step(main_goal)

        # combine up and down vecs for final score
        for goal in graph:
            goal.final_score = updown.final_agg(goal)


        # if len(graph) > 3:
        #     print ("goal graph after down: \n")
        #     main_goal._print_with_scores()
            # print (f"\n raw scores: {tmp1}")
            # print (f"\n agg scores: {tmp2}")
            # print (f"\n context scores: {tmp4}")

        goal_scores = [goal.final_score.squeeze(0) for goal in graph]
        # print("shape \n")
        # print (graph[0].final_score.shape)
        goal_scores = torch.cat(goal_scores, dim=0)

        goal_probs = F.softmax(goal_scores, dim=0)
        # print (f"goal prob shape {goal_probs.shape}")
        # print (f"goal probs {goal_probs}")
        # goal_probs = torch.FloatTensor(goal_scores)

        goal_m = Categorical(goal_probs)

        # index for best node
        goal_idx = goal_m.sample()
        # print (f"goal_idx {goal_idx}")



        goal_prob = goal_m.log_prob(goal_idx)

    except Exception as e:
        # print (" bs" + 80 * "!")
        # print (f"reps {representations}")
        # print (f"scores {scores}")
        # print (f"goal scores {goal_scores}")
        # print (f"goal probs {goal_probs}")
        print ("goal selection error: \n")
        traceback.print_exc()

    return graph[goal_idx], representations[goal_idx], goal_prob
    # return graph[goal], representations[goal]


'''

Torch implementation of TacticZero with GNN encoder and random Induct term selection 

'''
class GNNVanilla(Agent):
    def __init__(self, tactic_pool, replay_dir = None, train_mode = True):
        super().__init__(tactic_pool)

        self.ARG_LEN = 5
        self.train_mode = train_mode
        self.context_rate = 5e-5
        self.tac_rate = 5e-5
        self.arg_rate = 5e-5
        self.term_rate = 5e-5


        self.embedding_dim = 256
        self.gamma = 0.99 # 0.9

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # updown = UpDown=AggSiblings(device), agg_children=AggChildren(), agg_context=AggContext(device), device=device)
        self.updown = UpDownVec(self.device, 128).to(self.device)
        self.score_net = utp_model.ScorePolicy(128).to(self.device)
        # self.context_net = utp_model.ContextPolicy().to(self.device)

        self.tac_net = utp_model.TacPolicy(len(tactic_pool)).to(self.device)
        self.arg_net = utp_model.ArgPolicy(len(tactic_pool), self.embedding_dim).to(self.device)
        self.term_net = utp_model.TermPolicy(len(tactic_pool), self.embedding_dim).to(self.device)

        self.induct_gnn = inner_embedding_network.message_passing_gnn_induct(1000, self.embedding_dim // 2, num_iterations=2, device=self.device)
        self.optimizer_induct = torch.optim.RMSprop(list(self.induct_gnn.parameters()), lr = self.term_rate)

        self.optimizer_updown = torch.optim.RMSprop(list(self.updown.parameters()), lr = self.context_rate)
        # self.optimizer_context = torch.optim.RMSprop(list(self.context_net.parameters()), lr=self.context_rate)
        self.optimizer_score = torch.optim.RMSprop(list(self.score_net.parameters()), lr=self.context_rate)
        self.optimizer_tac = torch.optim.RMSprop(list(self.tac_net.parameters()), lr=self.tac_rate)
        self.optimizer_arg = torch.optim.RMSprop(list(self.arg_net.parameters()), lr=self.arg_rate)
        self.optimizer_term = torch.optim.RMSprop(list(self.term_net.parameters()), lr=self.term_rate)


        if replay_dir:
            # with open(replay_dir) as f:
            #     self.replays = json.load(f)

            with open(replay_dir, "rb") as f:
                self.replays = pickle.load(f)
        else:
            self.replays = {}

        self.optimizer_encoder_premise = torch.optim.RMSprop(list(self.encoder_premise.parameters()), lr=self.term_rate)
        self.optimizer_encoder_goal = torch.optim.RMSprop(list(self.encoder_goal.parameters()), lr=self.term_rate)

    def load_encoder(self):
        self.encoder_premise = torch.load("model_checkpoints/gnn_encoder_latest_premise")
        self.encoder_goal = torch.load("model_checkpoints/gnn_encoder_latest_goal")
        return

    def save(self):
        # torch.save(self.context_net, "model_checkpoints/gnn_new_goal_context")
        torch.save(self.score_net, "model_checkpoints/gnn_updown_vec_score")

        torch.save(self.updown, "model_checkpoints/gnn_updown_vec_updown")

        torch.save(self.tac_net, "model_checkpoints/gnn_up_down_vec_tac")
        torch.save(self.arg_net, "model_checkpoints/gnn_up_down_vec_arg")
        torch.save(self.term_net, "model_checkpoints/gnn_up_down_vec_term")
        torch.save(self.induct_gnn, "model_checkpoints/gnn_up_down_vec_induct")
        torch.save(self.encoder_premise, "model_checkpoints/gnn_up_down_vec_encoder_premise_e2e")
        torch.save(self.encoder_goal, "model_checkpoints/gnn_up_down_vec_encoder_goal_e2e")

        
    
    def load(self):
        self.score_net = torch.load("model_checkpoints/gnn_updown_vec_score")
        self.updown = torch.load("model_checkpoints/gnn_updown_vec_updown")

        self.tac_net = torch.load("model_checkpoints/gnn_up_down_vec_tac")
        self.arg_net = torch.load("model_checkpoints/gnn_up_down_vec_arg")
        self.term_net = torch.load("model_checkpoints/gnn_up_down_vec_term")
        self.induct_gnn = torch.load("model_checkpoints/gnn_up_down_vec_induct")

        self.encoder_premise = torch.load("model_checkpoints/gnn_up_down_vec_encoder_premise_e2e")
        self.encoder_goal = torch.load("model_checkpoints/gnn_up_down_vec_encoder_goal_e2e")

        self.optimizer_updown = torch.optim.RMSprop(list(self.updown.parameters()), lr = self.context_rate)
        self.optimizer_score = torch.optim.RMSprop(list(self.score_net.parameters()), lr=self.context_rate)
        self.optimizer_tac = torch.optim.RMSprop(list(self.tac_net.parameters()), lr=self.tac_rate)
        self.optimizer_arg = torch.optim.RMSprop(list(self.arg_net.parameters()), lr=self.arg_rate)
        self.optimizer_term = torch.optim.RMSprop(list(self.term_net.parameters()), lr=self.term_rate)
        self.optimizer_induct = torch.optim.RMSprop(list(self.induct_gnn.parameters()), lr = self.term_rate)
        self.optimizer_encoder_premise = torch.optim.RMSprop(list(self.encoder_premise.parameters()), lr=self.term_rate)
        self.optimizer_encoder_goal = torch.optim.RMSprop(list(self.encoder_goal.parameters()), lr=self.term_rate)

    def run(self, env, allowed_fact_batch, allowed_arguments_ids, candidate_args, max_steps=50):
        
        allowed_fact_batch = allowed_fact_batch.to(self.device)
        # encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
        fringe_pool = []
        tac_pool = []
        arg_pool = []
        action_pool = []
        reward_pool = []
        reward_print = []
        tac_print = []
        induct_arg = []
        proved = 0
        iteration_rewards = []
        steps = 0        
        replay_flag = False

        trace = []
        
        start_t = time.time()
        for t in range(max_steps):
            # print ("Tree: ")
            # env.graph._print()
            # env.graph._print()
            # print ("Current goals: ")
            # print ([g.goal["plain"]["goal"] for g in env.current_goals])
            # print ("Graph print: ")
            # print (env.graph)
            # env.graph._print()
            # print (env.current_goals)
            # print ([g.goal for g in env.current_goals])
            # gather all the goals in the history using goal encoder
            try:
                #intialise scores for graph using score_net
                target_goal_node, target_representation, goal_prob = embed_goal_graph(env.graph, env.current_goals, self.encoder_goal, self.score_net, self.updown)
            except Exception as e:
                print ("Goal selection error {}".format(e))
                print (f"env graph \n")
                env.graph._print()
                print (f"env goals {[g.goal for g in env.current_goals]}")
                traceback.print_exc()
                return ("Encoder error", str(e))


            fringe_pool.append(goal_prob)

            target_goal = target_goal_node.goal['polished']['goal']


            # target_goal = env.current_goals[goal].goal['polished']['goal']
            # target_representation = representations[goal]

            tac_input = target_representation
            tac_input = tac_input.to(self.device)

            tac_probs = self.tac_net(tac_input)
            tac_m = Categorical(tac_probs)
            tac = tac_m.sample()
            tac_pool.append(tac_m.log_prob(tac))
            action_pool.append(tactic_pool[tac])
            tac_print.append(tac_probs.detach())


            tac_tensor = tac.to(self.device)

            if tactic_pool[tac] in no_arg_tactic:
                tactic = tactic_pool[tac]
                arg_probs = []
                arg_probs.append(torch.tensor(0))
                arg_pool.append(arg_probs)
                
            elif tactic_pool[tac] == "Induct_on":

                target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal))


                arg_probs = []
                candidates = []

                # tokens = target_goal.split()
                # tokens = list(dict.fromkeys(tokens))
                #

                tokens = [[t] for t in target_graph.labels if t[0] == "V"]

                token_inds = [i for i,t in enumerate(target_graph.labels) if t[0] == "V"]

                if tokens:



                    # pass whole graph through Induct GNN
                    induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal))

                    induct_nodes = self.induct_gnn(induct_graph.x.to(self.device), induct_graph.edge_index.to(self.device))



                    # select representations of Variable nodes nodes with ('V' label only)

                    token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))

                    # pass through term_net as before



                    target_representation_list = [target_representation for _ in tokens]

                    target_representations = torch.cat(target_representation_list)

                    candidates = torch.cat([token_representations, target_representations], dim=1)
                    candidates = candidates.to(self.device)


                    scores = self.term_net(candidates, tac_tensor)
                    term_probs = F.softmax(scores, dim=0)
                    try:
                        term_m = Categorical(term_probs.squeeze(1))
                    except:
                        print("probs: {}".format(term_probs))
                        print("candidates: {}".format(candidates.shape))
                        print("scores: {}".format(scores))
                        print("tokens: {}".format(tokens))
                        exit()

                    term = term_m.sample()

                    arg_probs.append(term_m.log_prob(term))

                    induct_arg.append(tokens[term])
                    tm = tokens[term][0][1:] # remove headers, e.g., "V" / "C" / ...
                    arg_pool.append(arg_probs)
                    if tm:
                        tactic = "Induct_on `{}`".format(tm)
                        #print (tactic)
                    else:
                        print("tm is empty")
                        print(tokens)
                        # only to raise an error
                        tactic = "Induct_on"
                else:
                    arg_probs.append(torch.tensor(0))
                    induct_arg.append("No variables")

                    arg_pool.append(arg_probs)
                    tactic = "Induct_on"
            else:
                hidden0 = hidden1 = target_representation#unsqueeze(0)
                hidden0 = hidden0.to(self.device)
                hidden1 = hidden1.to(self.device)

                hidden = (hidden0, hidden1)
                
                # concatenate the candidates with hidden states.

                hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]
                
                try:
                    hiddenl = torch.cat(hiddenl)
                except Exception as e:
                    return ("hiddenl error...{}", str(e))

                #encode premises with premise GNN
                #encoded_fact_pool = self.encoder_premise.forward(allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                #encode and pool for digae
                #todo this at start to avoid recomputation?
                encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device), allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                candidates = candidates.to(self.device)
                            
                input = tac_tensor
                # print (input.shape, candidates.shape)#, hidden.shape)
                # run it once before predicting the first argument
                hidden, _ = self.arg_net(input, candidates, hidden)

                # the indices of chosen args
                arg_step = []
                arg_step_probs = []
                
                if tactic_pool[tac] in thm_tactic:
                    arg_len = 1
                else:
                    arg_len = self.ARG_LEN#ARG_LEN

                for _ in range(arg_len):
                    hidden, scores = self.arg_net(input, candidates, hidden)
                    arg_probs = F.softmax(scores, dim=0)
                    arg_m = Categorical(arg_probs.squeeze(1))
                    arg = arg_m.sample()
                    arg_step.append(arg)
                    arg_step_probs.append(arg_m.log_prob(arg))

                    hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
                    hidden1 = hidden[1].squeeze().repeat(1, 1, 1)
                    
                    # encoded chosen argument
                    input = encoded_fact_pool[arg].unsqueeze(0)#.unsqueeze(0)

                    # renew candidates                
                    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    hiddenl = torch.cat(hiddenl)
                    #appends both hidden and cell states (when paper only does hidden?)
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(self.device)

                arg_pool.append(arg_step_probs)

                tac = tactic_pool[tac]
                arg = [candidate_args[j] for j in arg_step]


                tactic = env.assemble_tactic(tac, arg)

            action = (target_goal_node, tactic)
            
            trace.append(action)

            # print (f"stepping with {action}")

            # reward, done = env.step(action)
            try:
                # print (f"Stepping with {target_goal_node.goal['plain']}, {tactic}")
                # print (f"Len current goals: {len(env.current_goals)}")
                # print (f"Unique len current goals {[g.goal['plain'] for g in env.current_goals]}")
                reward, done = env.step(action)


            except Exception as e:
                print("Step exception raised.")
                print (e)
                return ("Step error", action)
                # print("Fringe: {}".format(env.history))
                print("andling: {}".format(env.handling))
                print("Using: {}".format(env.using))
                # try again
                # counter = env.counter
                frequency = env.frequency
                env.close()
                print("Aborting current game ...")
                print("Restarting environment ...")
                print(env.goal)
                env = HolEnv(env.goal)
                flag = False
                break

            if t == max_steps - 1:
                reward = -5
                
            #could add environment state, but would grow rapidly
            trace.append((reward, action))
            
            reward_print.append(reward)
            reward_pool.append(reward)

            steps += 1
            total_reward = float(np.sum(reward_print))

            if done == True:
                print ("Goal Proved in {} steps".format(t+1))
                iteration_rewards.append(total_reward)

                map = construct_new_map(env.history, env.action_history)
                g_ = env.goal
                #
                try:
                    proof_len, proof = find_best_proof(g_, map)
                except Exception as e:


                    # todo store cases in file
                    # todo when arg enclosed in '', just filter it out?

                    print (f"Proof validation exception {e}")
                    print ("\n\n\n\n")
                    break

                    # print ("history: ")
                    # for i in range(1, len(env.history)):
                    #     action = env.action_history[i - 1]
                    #     goals = env.history[i]
                    #     new_goals = [g for g in env.history[i] if g not in env.history[i-1]]
                    #
                    #
                    #     print (f"action: {action}")
                    #     print (f"new goals {new_goals}")
                    #     print (f"goals {goals}")
                    #
                    # graphs = graph_from_history(env.history, env.action_history)
                    # for graph in graphs:
                    #     print ("graph: \n")
                    #     graph._print()
                    #
                    #
                    # traceback.print_exc()

                # print (proof_len, proof)
                # # print (g_)
                #
                print (f"Script: {proof}")
                #
                data = env.query(g_, proof, False)
                print (f"data {data}")

                if data != []:
                    print ("replay error" + 40 * "!!")
                    # todo store cases in file

                    # todo break here if not valid?
                    break


                # print ("Graphs \n")
                # graphs = graph_from_history(env.history, env.action_history)
                # for graph in graphs:
                #     print ("Graph ")
                    # print (graph.children, graph.parent, graph.goal)
                    # graph._print()

                #if proved, add to successful replays for this goal
                if env.goal in self.replays.keys():

                    #if proof done in less steps than before, add to dict
                    # if steps < len(self.replays[env.goal][0]):
                    #     print ("adding to replay")
                    #     # print (env.history)
                    #     self.replays[env.goal] = (env.history, env.action_history, reward_pool, (fringe_pool, tac_pool, arg_pool))

                    self.replays[env.goal].append((env.history, env.action_history, reward_pool, (torch.stack(fringe_pool).cpu().detach().numpy(), torch.stack(tac_pool).cpu().detach().numpy(), [torch.stack(arg).cpu().detach().numpy() for arg in arg_pool])))

                else:

                    print ("Initial add to db...")
                    # print (env.history)
                    if env.history is not None:
                        self.replays[env.goal] = [(env.history, env.action_history, reward_pool, (torch.stack(fringe_pool).cpu().detach().numpy(), torch.stack(tac_pool).cpu().detach().numpy(), [torch.stack(arg).cpu().detach().numpy() for arg in arg_pool]))]

                    else:
                        print ("history is none.............")
                        # print (env.history)
                        print (env)
                break

            if t == max_steps - 1:
                print("Failed.")

                if env.goal in self.replays.keys():
                    replay_flag = True

                    return trace, steps, done, 0, 0, replay_flag

                #print("Rewards: {}".format(reward_print))
                # print("Rewards: {}".format(reward_pool))
                #print("Tactics: {}".format(action_pool))
                # print("Mean reward: {}\n".format(np.mean(reward_pool)))
                #print("Total: {}".format(total_reward))
                iteration_rewards.append(total_reward)


        if self.train_mode:

            self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)
        
        return trace, steps, done, 0, float(np.sum(reward_print)), replay_flag



    def update_params(self, reward_pool, fringe_pool, arg_pool, tac_pool, step_count):
        # Update policy
        # Discount reward
        print("Updating parameters ... ")
        running_add = 0
        for i in reversed(range(step_count)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        self.optimizer_score.zero_grad()
        self.optimizer_updown.zero_grad()
        self.optimizer_tac.zero_grad()
        self.optimizer_arg.zero_grad()
        self.optimizer_term.zero_grad()
        self.optimizer_induct.zero_grad()
        self.optimizer_encoder_premise.zero_grad()
        self.optimizer_encoder_goal.zero_grad()


        total_loss = 0

        for i in range(step_count):
            reward = reward_pool[i]
            
            fringe_loss = -fringe_pool[i] * (reward)

            # print ("Trying graph: \n\n")
            # g = make_dot(fringe_loss)
            # g.view()

            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)

            tac_loss = -tac_pool[i] * (reward)

            # if i == 0:
            #     print ("Trying graph: \n\n")
            #     g = make_dot(arg_loss)
            #     g.view()

            loss = fringe_loss + tac_loss + arg_loss
            total_loss += loss


        # print ("Trying graph: \n\n")
        # g = make_dot(fringe_loss)
        # g.view()

        total_loss.backward()


        self.optimizer_score.step()
        self.optimizer_updown.step()
        self.optimizer_tac.step()
        self.optimizer_arg.step()
        self.optimizer_term.step()
        self.optimizer_induct.step()
        self.optimizer_encoder_premise.step()
        self.optimizer_encoder_goal.step()
        return

    #redo with new goal setup


    def replay_known_proof(self, env, allowed_fact_batch, allowed_arguments_ids, candidate_args):
        #known_history = random.sample(self.replays[env.goal][1], 1)[0]#[0]

        # always replay shortest found proof for now
        reps = self.replays[env.goal]
        rep_lens = [len(rep[0]) for rep in reps]
        min_rep = reps[rep_lens.index(min(rep_lens))]

        known_history, known_action_history, reward_history, _ = min_rep

        # known_history, known_action_history, reward_history, _ = self.replays[env.goal][0]

        # print ("hist lens")
        # print (len(known_history, len(known_action_history), len(reward_history)))
        try:
            allowed_fact_batch = allowed_fact_batch.to(self.device)
            fringe_pool = []
            tac_pool = []
            arg_pool = []
            action_pool = []
            reward_pool = []
            reward_print = []
            tac_print = []
            induct_arg = []
            iteration_rewards = []
            steps = 0




            hist_graphs = graph_from_history(known_history, known_action_history)


            for t in range(len(hist_graphs)):
                # true_resulting_fringe = known_history[t + 1]

                # graph_t = known_history[t]

                graph_t = hist_graphs[t]

                nodes_t = graph_env.nodes_list(graph_t, result=[])

                goals_t = [g.goal for g in nodes_t]

                chosen_goal, tactic = known_action_history[t]

                node_idx = goals_t.index(chosen_goal)
                chosen_node = nodes_t[node_idx]


                try:
                    #intialise scores for graph using score_net
                    target_representation, chosen_prob =  embed_goal_graph_replay(graph_t, nodes_t, self.encoder_goal, self.score_net, chosen_node, self.updown)
                except Exception as e:
                    print ("Goal selection error {}".format(e))
                    traceback.print_exc()
                    return ("Encoder error", str(e))


                fringe_pool.append(chosen_prob)

                target_goal = chosen_node.goal['polished']['goal']



                tac_input = target_representation.unsqueeze(0)
                tac_input = tac_input.to(self.device)

                tac_probs = self.tac_net(tac_input)
                tac_m = Categorical(tac_probs)

                true_tactic_text = tactic #true_resulting_fringe["by_tactic"]
                # print (true_tactic_text)

                if true_tactic_text in no_arg_tactic:
                    true_tac_text = true_tactic_text
                else:
                    # true_tactic_text = "Induct_on `ll`"
                    tac_args = re.findall(r'(.*?)\[(.*?)\]', true_tactic_text)
                    tac_term = re.findall(r'(.*?) `(.*?)`', true_tactic_text)
                    tac_arg = re.findall(r'(.*?) (.*)', true_tactic_text)

                    if tac_args:
                        true_tac_text = tac_args[0][0]
                        true_args_text = tac_args[0][1].split(", ")
                    elif tac_term:  # order matters # TODO: make it irrelavant
                        true_tac_text = tac_term[0][0]
                        true_args_text = tac_term[0][1]
                    elif tac_arg:  # order matters because tac_arg could match () ``
                        true_tac_text = tac_arg[0][0]
                        true_args_text = tac_arg[0][1]
                    else:
                        true_tac_text = true_tactic_text


                true_tac = torch.tensor([tactic_pool.index(true_tac_text)])
                true_tac = true_tac.to(self.device)

                tac_pool.append(tac_m.log_prob(true_tac))
                action_pool.append(tactic_pool[true_tac])
                tac_print.append(tac_probs.detach())


                tac_tensor = true_tac.to(self.device)

                assert tactic_pool[true_tac.item()] == true_tac_text




                if tactic_pool[true_tac] in no_arg_tactic:
                    # print ("no arg")
                    tactic = tactic_pool[true_tac]
                    arg_probs = []
                    arg_probs.append(torch.tensor(0))
                    arg_pool.append(arg_probs)


                elif tactic_pool[true_tac] == "Induct_on":
                    # print ("induct")
                    #TODO need to find a new way to do this...
                    #Could pass the whole Graph for the expression then take the embeddings for the specific nodes corresponding to VAR
                    #Then pass the embeddings to a new term net to softmax and select one. Need a way to map from VAR in graph back to the variable

                    arg_probs = []
                    candidates = []

                    target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal))


                    tokens = [[t] for t in target_graph.labels if t[0] == "V"]

                    token_inds = [i for i,t in enumerate(target_graph.labels) if t[0] == "V"]

                    if tokens:

                        # print ("replaying induction")

                        true_term = torch.tensor([tokens.index(["V" + true_args_text])])

                        true_term = true_term.to(device)

                        #feed through induct GNN to get representation

                        induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal))

                        induct_nodes = self.induct_gnn(induct_graph.x.to(self.device), induct_graph.edge_index.to(self.device))

                        # select representations of Variable nodes nodes with ('V' label only)

                        token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))

                        # pass through term_net as before



                        target_representation_list = [target_representation for _ in tokens]

                        target_representations = torch.cat(target_representation_list)

                        candidates = torch.cat([token_representations, target_representations], dim=1)
                        candidates = candidates.to(self.device)


                        scores = self.term_net(candidates, tac_tensor)
                        term_probs = F.softmax(scores, dim=0)

                        try:
                            term_m = Categorical(term_probs.squeeze(1))
                        except:
                            print("probs: {}".format(term_probs))
                            print("candidates: {}".format(candidates.shape))
                            print("scores: {}".format(scores))
                            print("tokens: {}".format(tokens))
                            exit()

                        arg_probs.append(term_m.log_prob(true_term))

                        induct_arg.append(tokens[true_term])
                        tm = tokens[true_term][0][1:] # remove headers, e.g., "V" / "C" / ...
                        arg_pool.append(arg_probs)

                        assert tm == true_args_text

                        if tm:
                            tactic = "Induct_on `{}`".format(tm)
                        else:
                            print("tm is empty")
                            print(tokens)
                            # only to raise an error
                            tactic = "Induct_on"
                    else:
                        arg_probs.append(torch.tensor(0))
                        induct_arg.append("No variables")
                        arg_pool.append(arg_probs)
                        tactic = "Induct_on"



                else:
                    # print ("arg")
                    hidden0 = hidden1 = target_representation
                    hidden0 = hidden0.to(self.device)
                    hidden1 = hidden1.to(self.device)

                    hidden = (hidden0, hidden1)

                    # concatenate the candidates with hidden states.


                    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                    hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                    try:
                        hiddenl = torch.cat(hiddenl)
                    except Exception as e:
                        return ("hiddenl error...{}", str(e))


                    encoded_fact_pool = self.encoder_premise.encode_and_pool(allowed_fact_batch.x.to(device), allowed_fact_batch.x.to(device),allowed_fact_batch.edge_index.to(device), allowed_fact_batch.batch.to(device))
                    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                    candidates = candidates.to(self.device)

                    input = tac_tensor

                    # run it once before predicting the first argument
                    hidden, _ = self.arg_net(input, candidates, hidden)

                    # the indices of chosen args
                    arg_step = []
                    arg_step_probs = []

                    if tactic_pool[true_tac] in thm_tactic:
                        arg_len = 1
                    else:
                        arg_len = self.ARG_LEN#ARG_LEN


                    for i in range(arg_len):
                        hidden, scores = self.arg_net(input, candidates, hidden)
                        arg_probs = F.softmax(scores, dim=0)
                        arg_m = Categorical(arg_probs.squeeze(1))

                        if isinstance(true_args_text, list):
                            try:
                                name_parser = true_args_text[i].split(".")
                            except:
                                print (i)
                                print (true_args_text)
                                print (known_history)
                                exit()
                            theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                            theorem_name = name_parser[1]
                            #todo not sure if reverse_database will work...

                            # true_arg_exp = reverse_database[(theory_name, theorem_name)]
                            true_arg_exp = reverse_database[(theory_name.strip("\'").strip("\""), theorem_name.strip("\'").strip("\""))]
                        else:
                            name_parser = true_args_text.split(".")
                            theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                            theorem_name = name_parser[1]
                            true_arg_exp = reverse_database[(theory_name.strip("\'").strip("\""), theorem_name.strip("\'").strip("\""))]
                        true_arg = torch.tensor(candidate_args.index(true_arg_exp))
                        true_arg = true_arg.to(self.device)



                        arg_step.append(true_arg)
                        arg_step_probs.append(arg_m.log_prob(true_arg))

                        hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
                        hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

                        # encoded chosen argument
                        input = encoded_fact_pool[true_arg].unsqueeze(0)#.unsqueeze(0)

                        # renew candidates
                        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
                        hiddenl = [hc.unsqueeze(0) for _ in allowed_arguments_ids]

                        hiddenl = torch.cat(hiddenl)
                        #appends both hidden and cell states (when paper only does hidden?)
                        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
                        candidates = candidates.to(self.device)

                    arg_pool.append(arg_step_probs)



                try:
                    reward = reward_history[t]
                    #done = t + 2 == len(known_history)
                except:
                    print("Step exception raised.")
                    return ("Step error", action)
                    # print("Fringe: {}".format(env.history))
                    print("Handling: {}".format(env.handling))
                    print("Using: {}".format(env.using))
                    # try again
                    # counter = env.counter
                    frequency = env.frequency
                    env.close()
                    print("Aborting current game ...")
                    print("Restarting environment ...")
                    print(env.goal)
                    env = HolEnv(env.goal)
                    flag = False
                    break

                reward_print.append(reward)
                reward_pool.append(reward)
                steps += 1

            total_reward = float(np.sum(reward_print))

        except Exception as e:
            print (f'Replay error {str(e)}')

            # remove erroneous replay from buffer

            print ("removing replay")
            reps.remove(min_rep)
            assert min_rep not in reps

            if len(self.replays[env.goal]) == 0:
                del self.replays[env.goal]
            return

        if self.train_mode:
            self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)

        return

    def save_replays(self):
        # with open("gnn_agent_new_goal_replays.json", "w") as f:
        #     json.dump(self.replays, f)

        with open("old_replays/gnn_agent_up_down_vec_replays.pk", "wb") as f:
            pickle.dump(self.replays, f)


class Experiment_GNN:
    def __init__(self, agent, goals, database, num_iterations, train_mode= True):
        self.agent = agent
        self.train_mode = train_mode
        self.goals = goals
        self.num_iterations = num_iterations
        self.database = database

    def train(self):
        env = HolEnv("T")
        env_errors = []
        agent_errors = []
        full_trace = []
        iter_times = []
        reward_trace = []
        proved_trace = []

        for iteration in range(self.num_iterations):
            iter_trace = {}
            it_start = time.time()

            prove_count = 0

            for i, goal in enumerate(self.goals):
                print ("Goal #{}".format(str(i+1)))

                try:
                    env.reset(goal[1])
                except Exception as e:
                    print ("Restarting environment..")
                    env = HolEnv("T")
                    continue
                 
                try:
                    allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(env, goal)
                except Exception as e:
                    print ("Env error: {}".format(e))
                    #env_errors.append((goal, "Error generating fact pool", i))
                    continue
                    
                result = self.agent.run(env, 
                                        allowed_fact_batch,
                                        allowed_arguments_ids, candidate_args,  max_steps=50)

                
                #agent run returns (error_msg, details) if error, larger tuple otherwise
                if len(result) == 2:
                    #agent_errors.append((result, i))
                    continue

                else:
                    trace, steps, done, goal_time, reward_total, replay_flag = result
                    #iter_trace[goal[0]] = (trace, steps, done, goal_time, reward_total)



                    if replay_flag:
                        print ("replaying proof...")
                        #todo reset env?
                        #reset environment before replay
                        try:
                            env.reset(goal[1])
                        except Exception as e:
                            print ("Restarting environment..")
                            env = HolEnv("T")
                            continue

                        print ("env reset...")

                        try:
                            self.agent.replay_known_proof(env, allowed_fact_batch, allowed_arguments_ids, candidate_args)
                        except Exception as e:
                            traceback.print_exc()
                            print (f"replay error {e}")


                    reward_trace.append(reward_total)
                    if done:
                        prove_count += 1

            if self.train_mode:
                self.agent.save_replays()
                self.agent.save()
            #full_trace.append(iter_trace)
            #iter_times.append(time.time() - it_start)
            proved_trace.append(prove_count)

            if self.train_mode:
                date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')


                # with open(f"traces/graph_goal/gnn_graph_goal_agent_reward_trace_{date}.pk", "wb") as f:
                #     pickle.dump(reward_trace, f)
                #
                # with open("traces/graph_goal/gnn_graph_agent_model_errors.pk", "wb") as f:
                #     pickle.dump((env_errors, agent_errors), f)

                with open(f"traces/graph_goal/gnn_updown_vec_proved_{date}.pk", "wb") as f:
                    pickle.dump(proved_trace, f)


                #save parameters every iteration
                self.agent.save()

            print (f"done {prove_count}")
        return #full_trace, env_errors, agent_errors, iter_times
            
                
                
    def load_encoded_db(self, encoded_db_dir):
        self.encoded_database = torch.load(encoded_db_dir)

    def load_db(self, db_dir):
        with open(db_dir) as f:
            self.database = json.load(f)

    def gen_fact_pool(self, env, goal):

        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', goal[0])))
        goal_theory = self.database[goal[0]][0]

        # polished_goal = env.current_goals[0].goal["polished"]["goal"] #env.fringe["content"][0]["polished"]["goal"]
        polished_goal = goal[0]
        try:
            allowed_arguments_ids = []
            candidate_args = []
            for i,t in enumerate(self.database):
                if self.database[t][0] in allowed_theories and (self.database[t][0] != goal_theory or int(self.database[t][2]) < int(self.database[polished_goal][2])):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)

            env.toggle_simpset("diminish", goal_theory)
            #print("Removed simpset of {}".format(goal_theory))

        except:
            allowed_arguments_ids = []
            # candidate_args = []
            # for i,t in enumerate(self.database):
            #     if self.database[t][0] in allowed_theories:
            #         allowed_arguments_ids.append(i)
            #         candidate_args.append(t)
            raise Exception("Theorem not found in database.")


        graphs = [graph_db[t] for t in candidate_args]
        
        loader = DataLoader(graphs, batch_size = len(candidate_args))
        
        allowed_fact_batch = next(iter(loader))


        # encoded_fact_pool = gnn_enc.forward(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))

        # return encoded_fact_pool, allowed_arguments_ids, candidate_args
        #return batch with graphs to encode by agent
        return allowed_fact_batch, allowed_arguments_ids, candidate_args




def run_experiment():

    try:
        agent = GNNVanilla(tactic_pool, train_mode=True, replay_dir="old_replays/gnn_agent_up_down_vec_replays.pk")

        agent.load()

        exp_gnn = Experiment_GNN(agent, train_goals, compat_db, 1000, train_mode=True)

        exp_gnn.train()

    except Exception as e:
        print (f"Fatal error {e}")
        traceback.print_exc()
        run_experiment()

def run_test():
    try:
        agent = GNNVanilla(tactic_pool, replay_dir=None, train_mode=False)

        agent.load()

        exp_gnn = Experiment_GNN(agent, test_goals, compat_db, 1, train_mode=False)

        exp_gnn.train()

    except Exception as e:
        print (f"Fatal error {e}")
        traceback.print_exc()
        # run_experiment()

# todo want to set up dataset which stores full graph with proof history, so we can use this to train action selection separately
# todo Will result in dataset of graphs, with subgoals and goals which are proven, and associated tactics
# todo Can also use this to train tactic and argument networks offline, especially with stored log probs for importance sampling correction
# todo may even want separate GNN for embedding goals before action selection, then another GNN for goals for tactic/premise selection

# run_test()

run_experiment()
# run_experiment()


#sanity check encodings are similar between non-deterministic runs
# print ("history: ")s = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

# x = []
# for i,k in enumerate(compat_db.keys()):
#     test = torch_graph_dict[k]
#     x.append(cs(gnn_enc.forward(test.x.to(device), test.edge_index.to(device)), encoded_graph_db[i]).cpu().detach().numpy())
#     #assert gnn_enc.forward(test.x.to(device), test.edge_index.to(device))[0][0] == encoded_graph_db[i][0][0]

# np.mean(x)

