from torch_geometric import utils
import networkx as nx
import torch
import torch_geometric
import torch_geometric.utils as utils

from data.holist.process_holist import sexpression_to_graph
import matplotlib.pyplot as plt
import graphviz


def plot_pyg_graph_with_graphviz(data, node_labels=None):
    G = utils.to_networkx(data, to_undirected=False, remove_self_loops=True)

    # Manually transfer edge attributes
    for i, (u, v) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
        G[u.item()][v.item()]['edge_attr'] = data.edge_attr[i].item()

    # edge_labels = {}
    # for u, v, data in G.edges(data=True):
    #     edge_labels[(u, v)] = data['edge_attr']

    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(size='6,6')
    for node, data in G.nodes(data=True):
        label = str(node_labels[node]) if node_labels else str(node)
        dot.node(str(node), label=label, fontsize='25')

    for u, v, data in G.edges(data=True):
        edge_label = str(data.get('edge_attr', ''))
        dot.edge(str(u), str(v), label=edge_label, fontsize='25')

    # dot.attr(fontsize='20')
    # Render and view graph
    dot.view()



if __name__ == '__main__':
    # sexpression ="(a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (c (fun (fun (cart (real) M) (fun (cart (real) M) (bool))) (bool)) gauge) (v (fun (cart (real) M) (fun (cart (real) M) (bool))) d))) (a (c (fun (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (bool)) (bool)) !) (l (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p1) (a (c (fun (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (bool)) (bool)) !) (l (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p2) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (fun (fun (cart (real) M) (bool)) (bool))) tagged_division_of) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p1)) (a (c (fun (list (prod (cart (real) M) (cart (real) M))) (fun (cart (real) M) (bool))) closed_interval) (a (a (c (fun (prod (cart (real) M) (cart (real) M)) (fun (list (prod (cart (real) M) (cart (real) M))) (list (prod (cart (real) M) (cart (real) M))))) CONS) (a (a (c (fun (cart (real) M) (fun (cart (real) M) (prod (cart (real) M) (cart (real) M)))) ,) (v (cart (real) M) a)) (v (cart (real) M) b'))) (c (list (prod (cart (real) M) (cart (real) M))) NIL))))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (fun (cart (real) M) (fun (cart (real) M) (bool))) (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (bool))) fine) (v (fun (cart (real) M) (fun (cart (real) M) (bool))) d)) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p1))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (fun (fun (cart (real) M) (bool)) (bool))) tagged_division_of) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p2)) (a (c (fun (list (prod (cart (real) M) (cart (real) M))) (fun (cart (real) M) (bool))) closed_interval) (a (a (c (fun (prod (cart (real) M) (cart (real) M)) (fun (list (prod (cart (real) M) (cart (real) M))) (list (prod (cart (real) M) (cart (real) M))))) CONS) (a (a (c (fun (cart (real) M) (fun (cart (real) M) (prod (cart (real) M) (cart (real) M)))) ,) (v (cart (real) M) a')) (v (cart (real) M) b))) (c (list (prod (cart (real) M) (cart (real) M))) NIL))))) (a (a (c (fun (fun (cart (real) M) (fun (cart (real) M) (bool))) (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (bool))) fine) (v (fun (cart (real) M) (fun (cart (real) M) (bool))) d)) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p2)))))) (a (a (c (fun (real) (fun (real) (bool))) real_lt) (a (c (fun (cart (real) N) (real)) vector_norm) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (cart (real) N))) vector_sub) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (cart (real) N))) vector_add) (a (a (c (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) (cart (real) N))) vsum) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p1)) (a (c (fun (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) (bool)) (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N))) GABS) (l (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) f) (a (c (fun (fun (cart (real) M) (bool)) (bool)) !) (l (v (cart (real) M) x) (a (c (fun (fun (fun (cart (real) M) (bool)) (bool)) (bool)) !) (l (v (fun (cart (real) M) (bool)) k) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) GEQ) (a (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) f) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (prod (cart (real) M) (fun (cart (real) M) (bool))))) ,) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) k)))) (a (a (c (fun (real) (fun (cart (real) N) (cart (real) N))) %) (a (c (fun (fun (cart (real) M) (bool)) (real)) content) (v (fun (cart (real) M) (bool)) k))) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x)))))))))))) (a (a (c (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) (cart (real) N))) vsum) (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (bool)) p2)) (a (c (fun (fun (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) (bool)) (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N))) GABS) (l (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) f) (a (c (fun (fun (cart (real) M) (bool)) (bool)) !) (l (v (cart (real) M) x) (a (c (fun (fun (fun (cart (real) M) (bool)) (bool)) (bool)) !) (l (v (fun (cart (real) M) (bool)) k) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) GEQ) (a (v (fun (prod (cart (real) M) (fun (cart (real) M) (bool))) (cart (real) N)) f) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (prod (cart (real) M) (fun (cart (real) M) (bool))))) ,) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) k)))) (a (a (c (fun (real) (fun (cart (real) N) (cart (real) N))) %) (a (c (fun (fun (cart (real) M) (bool)) (real)) content) (v (fun (cart (real) M) (bool)) k))) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x))))))))))))) (v (cart (real) N) y)))) (a (a (c (fun (real) (fun (real) (real))) real_div) (v (real) e)) (a (c (fun (num) (real)) real_of_num) (a (c (fun (num) (num)) NUMERAL) (a (c (fun (num) (num)) BIT0) (a (c (fun (num) (num)) BIT1) (c (num) _0)))))))))))))"
    #
    #
    # g = sexpression_to_graph(sexpression)
    #
    # x = torch.arange(len(g['tokens']))
    #
    # data = torch_geometric.data.Data(x=x, edge_index=torch.LongTensor(g['edge_index']),
    #                                  edge_attr=torch.LongTensor(g['edge_attr']))
    #
    # plot_pyg_graph_with_graphviz(data, g['tokens'])

    dot = graphviz.Digraph(format='pdf', engine='dot')

    '(a (a (c (fun (bool) (fun (bool) (bool))) =) (a (c (fun (bool) (bool)) ~) (a (c (fun (bool) (bool)) ~) (v (bool) b)))) (v (bool) b))'
    dot.edge('a', ' a ', label='1')
    dot.edge(' a ', 'c', label='1')
    dot.edge('c', 'fun', label='1')
    dot.edge('fun', 'bool', label='1')
    dot.edge('fun', ' fun ', label='2')
    dot.edge(' fun ', ' bool ', label='1')
    dot.edge(' fun ', '  bool  ', label='2')
    dot.edge('c', '=', label='2')
    dot.edge(' a ', '  a  ', label='2')
    dot.edge('  a  ', ' c ', label='1')
    dot.edge(' c ', '  fun  ', label='1')
    dot.edge('  fun  ', '  bool   ', label='1')
    dot.edge('  fun  ', '    bool    ', label='2')
    dot.edge(' c ', '~', label='2')
    dot.edge('  a  ', '   a   ', label='2')
    dot.edge('   a   ', '  c  ', label='1')
    dot.edge('  c  ', '   fun   ', label='1')
    dot.edge('   fun   ', '     bool     ', label='1')
    dot.edge('   fun   ', '      bool      ', label='2')
    dot.edge('  c  ',  '~ ', label='2')
    dot.edge(' a ', 'v', label='2')
    dot.edge('v', '       bool       ', label='1')
    dot.edge('v', 'b', label='2')
    dot.edge('a', ' v ', label='2')
    dot.edge(' v ', '        bool        ', label='1')
    dot.edge(' v ', ' b ', label='2')

    node_attr = {'fixedsize': 'true',
                 'width': '1',
                 'height': '1',
                 'fontsize': '25'}

    dot.node_attr = node_attr
    dot.edge_attr = {'fontsize': '25'}
    dot.view()

