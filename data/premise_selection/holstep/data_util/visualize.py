"Draw Larry's favorite graph!"
import sys

from graphviz import Digraph

sys.path.insert(0, 'data_util')


def draw(node_list, graph_name):
    dot = Digraph(comment=graph_name)
    for node in node_list:
        dot.node(str(node.id), node.name)
    for node in node_list:
        for out in node.outgoing:
            dot.edge(str(node.id),str(out.id))

    dot.render(graph_name, view=True)
