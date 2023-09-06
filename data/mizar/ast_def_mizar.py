import pickle
import numpy as np
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
from data.hol4.mongo_to_torch import get_depth_from_graph, get_directed_edge_index

class AST:
    def __init__(self, node, children=[], parent=None):
        self.node = node
        self.children = children
        self.parent = [parent]

    def _print(self, depth=1):
        print(depth * "--- " + self.node.value)
        if len(self.children) > 0:
            for child in self.children:
                child._print(depth + 1)

class Token:
    def __init__(self, value, type_, arity=None):
        self.value = value
        self.type_ = type_
        self.arity = arity

# assume ast has been passed with ast.node as function
def func_to_ast(ast, tokens, arity):
    if len(tokens) == 0:
        return ast

    node = tokens[0]
    tokens.pop(0)

    new_node = AST(node, children=[], parent=ast)

    if node.type_ == "variable":

        ast.children.append(new_node)

    elif node.type_ == "func" or node.type_ == "lambda":

        new_ast = func_to_ast(new_node, tokens, node.arity)
        ast.children.append(new_ast)

    if arity == 1:
        return ast
    else:
        return func_to_ast(ast, tokens, arity - 1)

def tokens_to_ast(tokens):
    ast = AST(tokens[0], children=[])
    tokens.pop(0)
    return func_to_ast(ast, tokens, ast.node.arity)

def polished_to_tokens_2(goal):
    polished_goal = [c for c in goal.split(" ") if c != '' and c != '\n']
    tokens = []


    while len(polished_goal) > 0:
        if polished_goal[0] == '*':
            polished_goal.pop(0)
            arity = 1

            while polished_goal[0] == '*':
                arity += 1
                polished_goal.pop(0)

            func = polished_goal[0]
            polished_goal.pop(0)

            # if func[0] == 'c':
            #     # should only be one string after the library
            #     func = func + "|" + polished_goal[0]
            #     polished_goal.pop(0)

            # otherwise variable func, and nothing following it

            tokens.append(Token(func, "func", arity))

        # variable or constant case
        else:
            var = polished_goal[0]
            polished_goal.pop(0)
            # lambda case
            if var == "/":
                tokens.append(Token("".join(var), "lambda", 2))
            else:
                # if var[0] == "C":
                #     # need to append this and the next as constants are space separated
                #     var = var + polished_goal[0]
                #     polished_goal.pop(0)

                tokens.append(Token("".join(var), "variable"))

        # print ([(tok.value, tok.type_, tok.arity) for tok in tokens])

    return tokens


def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


def print_graph(ast, path):
    G = nx.DiGraph()
    add_node(ast, G)

    labels = nx.get_node_attributes(G, 'value')
    pos = hierarchy_pos(G, ast.node)
    plt.figure(1, figsize=(15, 30))
    nx.draw(G, pos=pos, labels=labels, with_labels=True,
            arrowsize=20,
            node_color='none',
            node_size=6000)  # , font_weight='bold')

    labels = {e: G.get_edge_data(e[0], e[1])["child"] for e in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#FF0000")
    plt.savefig(path, format="PNG")


def add_node(ast, graph):
    graph.add_node(ast.node, value=str(ast.node.value))
    for i, child in enumerate(ast.children):
        graph.add_edge(ast.node, child.node, child=i)
        add_node(child, graph)

def add_lambda_children(lambda_ast):
    # child should be '|' with first child of that as variable, and rest as quantified scope
    var = lambda_ast.children[0].node.value
    sub_tree = lambda_ast.children[1]

    if sub_tree.node.value == var:
        return lambda_ast

    def apply_lambda(ast, var):
        if ast.node.value == var:
            lambda_ast.children.append(ast)
            ast.parent.append(lambda_ast)
            for child in ast.children:
                apply_lambda(child, var)
        else:
            for child in ast.children:
                apply_lambda(child, var)
        return

    apply_lambda(sub_tree, var)
    return lambda_ast

def process_lambdas(ast):
    ret = []

    def get_lambdas(ast):
        if ast.node.type_ == "lambda":
            ret.append(ast)
        for child in ast.children:
            get_lambdas(child)
        return
    get_lambdas(ast)
    for l in ret:
        add_lambda_children(l)

    return ast

def merge_leaves(ast):
    lambda_tokens = []

    # only merge leaf nodes if they're within the same quantified scope

    def run_lambdas(lambda_ast):
        var = lambda_ast.children[0].node.value

        # check for edge case lambda x: x
        if len(lambda_ast.children) == 1:
            return lambda_ast

        sub_tree = lambda_ast.children[1]
        lambda_token = lambda_ast.children[0]
        lambda_tokens.append(lambda_token)

        def merge_lambda(ast, var):
            # if lambda variable, and leaf node, point parents to original node
            if ast.node.value == var and ast.children == []:
                # this way ensures no duplicates
                for parent in ast.parent:
                    new_children = []
                    flag = False
                    for c in parent.children:
                        if c.node.value != var or c.children != []:
                            new_children.append(c)
                        elif flag == False:
                            new_children.append(lambda_token)
                            flag = True
                    parent.children = new_children

            for child in ast.children:
                merge_lambda(child, var)

            return ast

        merge_lambda(sub_tree, var)
        return lambda_ast

    def merge_all_lambdas(ast):
        ret = []

        def get_lambdas(ast):
            if ast.node.type_ == "lambda":
                ret.append(ast)
            for child in ast.children:
                get_lambdas(child)
            return

        get_lambdas(ast)

        for l in ret:
            run_lambdas(l)

        return ast

    merge_all_lambdas(ast)

    return ast

def rename(ast):
    if ast.node.value[0] == 'b':
        if ast.children != []:
            ast.node.value = "VARFUNC"
        else:
            ast.node.value = "VAR"

    for child in ast.children:
        rename(child)

    return ast



def goal_to_graph(polished_goal):
    # return rename(merge_leaves(process_lambdas(tokens_to_ast(polished_to_tokens_2(polished_goal)))))
    return rename(process_ast(polished_goal))

def goal_to_graph_labelled(polished_goal):
    return merge_leaves(process_lambdas(tokens_to_ast(polished_to_tokens_2(polished_goal))))

def nodes_list_to_senders_receivers(node_list):
    senders = []
    receivers = []
    for i, node in enumerate(node_list):
        for child in node.children:
            senders.append(i)
            receivers.append(node_list.index(child))
    return senders, receivers


unordered_ops = ['c&', 'c=', 'c<=>', 'c|']


def nodes_list_to_senders_receivers_labelled(node_list):
    senders = []
    receivers = []
    edge_labels = []
    for i, node in enumerate(node_list):
        for j, child in enumerate(node.children):
            senders.append(i)
            receivers.append(node_list.index(child))
            if node.node.value in unordered_ops:
                edge_labels.append(0)
            else:
                edge_labels.append(j)
    return senders, receivers, edge_labels

# def nodes_list_to_senders_receivers_labelled(node_list):
#     senders = []
#     receivers = []
#     edge_labels = []
#     for i, node in enumerate(node_list):
#         for j, child in enumerate(node.children):
#             senders.append(i)
#             receivers.append(node_list.index(child))
#             edge_labels.append(j)
#     return senders, receivers, edge_labels
#


def nodes_list(g, result=[]):
    result.append(g)

    for child in g.children:
        nodes_list(child, result)

    return list(set(result))






# with open("/home/sean/Documents/phd/aitp/data/hol4/graph_token_encoder.pk", "rb") as f:
#     token_enc = pickle.load(f)

def sp_to_torch(sparse):
    coo = sparse.tocoo()

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))  # .to_dense()


def graph_to_torch(g, token_enc=None):
    node_list = nodes_list(g, result=[])
    senders, receivers = nodes_list_to_senders_receivers(node_list)

    # get the one hot encoding from enc
    t_f = lambda x: np.array([x.node.value])

    node_features = list(map(t_f, node_list))

    node_features = token_enc.transform(node_features)

    edges = torch.tensor([senders, receivers], dtype=torch.long)

    nodes = sp_to_torch(node_features)

    return Data(x=nodes, edge_index=edges)

def graph_to_torch_labelled(g, token_enc=None):
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

    #old:
    # nodes = sp_to_torch(node_features)

    # new:
    # returning only the one-hot tensors
    coo = node_features.tocoo()
    nodes = torch.LongTensor(coo.col)

    return Data(x=nodes, edge_index=edges, edge_attr=torch.LongTensor(edge_labels), labels=labels)



def graph_to_dict(g):
    node_list = nodes_list(g, result=[])
    #    senders, receivers = nodes_list_to_senders_receivers(node_list)
    senders, receivers, edge_labels = nodes_list_to_senders_receivers_labelled(node_list)

    # define labels before renaming to keep original variables for induction
    # labels = [x.node.value for x in node_list]

    # rename variables to be constant
    for node in node_list:
        if node.node.value[0] == 'b':
            if node.children != []:
                node.node.value = "VARFUNC"
            else:
                node.node.value = "VAR"

    labels = [x.node.value for x in node_list]
    # t_f = lambda x: np.array([x.node.value])

    # node_features = list(map(t_f, node_list))

    return {'tokens': labels, 'edge_index': [senders, receivers], 'edge_attr': edge_labels}

    # attention_edge = get_directed_edge_index(len(labels), torch.LongTensor([senders, receivers]))
    # depth = get_depth_from_graph(len(labels), torch.LongTensor([senders, receivers]))
    #
    # return {'tokens': labels,
    #         'edge_index': [senders, receivers],
    #         'edge_attr': edge_labels,
    #         'attention_edge_index': attention_edge,
    #         'depth': depth
    #         }

    # return Data(x=nodes, edge_index=edges, edge_attr=torch.LongTensor(edge_labels), labels=labels)






'''

Add subexpression field to each node, to reduce graph size

'''
def ast_subexps(ast):
    val = ast.node.value
    child_exprs = ""
    for child in ast.children:
        child_exprs += ast_subexps(child)
    ast.subexp = val + child_exprs
    return val + child_exprs


'''

Share subexpressions

'''

def reduce_subexpressions(ast):
    nodes = nodes_list(ast, [])
    exprs = [node.subexp for node in nodes]

    dup_nodes = [node for node in nodes if exprs.count(node.subexp) > 1]

    dups = {}
    for node in dup_nodes:
        if node.subexp in dups:
            dups[node.subexp].append(node)
        else:
            dups[node.subexp] = [node]

    for nodes in dups.values():
        first_node = nodes[0]
        parents_list = [node.parent for node in nodes]


        for node in nodes[1:]:
            assert node not in first_node.parent
            # replace all references to same variable with first node
            for parent in node.parent:
                parent.children[parent.children.index(node)] = first_node



def process_ast(polished_goal):
    ast = tokens_to_ast(polished_to_tokens_2(polished_goal))
    ast_subexps(ast)
    reduce_subexpressions(ast)
    return ast
