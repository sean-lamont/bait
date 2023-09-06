import graphviz
import torch_geometric.utils as utils


def plot_pyg_graph_with_graphviz(data, node_labels=None):
    G = utils.to_networkx(data, to_undirected=False, remove_self_loops=True)

    # Manually transfer edge attributes
    for i, (u, v) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
        G[u.item()][v.item()]['edge_attr'] = data.edge_attr[i].item()

    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(size='6,6')
    for node, data in G.nodes(data=True):
        label = str(node_labels[node]) if node_labels else str(node)
        dot.node(str(node), label=label, fontsize='25')

    for u, v, data in G.edges(data=True):
        edge_label = str(data.get('edge_attr', ''))
        dot.edge(str(u), str(v), label=edge_label, fontsize='25')

    # Render and view graph
    dot.view()



if __name__ == '__main__':
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

