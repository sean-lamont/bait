from prover.search_tree_context import Status, ErrorNode, ProofFinishedNode
import pickle
import pygraphviz as pgv
from loguru import logger

def render_proof(path, filename=None):
    with open(path, 'rb') as f:
        trace = pickle.load(f)

    logger.info(f'Rendering proof of {trace.tree.goal}..')

    if not filename:
        filename = 'figures/' + trace.theorem.full_name + '.svg'

    siblings = []

    G = pgv.AGraph(name='root')

    def add_edges(node):
        if not node.out_edges:
            return
        for edge in node.out_edges:
            if any([d.status == Status.PROVED for d in edge.dst]):
                siblings_ = []
                for d in edge.dst:
                    if all([d_.status == Status.PROVED for d_ in edge.dst]):
                        if hasattr(edge.src, 'goal') and hasattr(d, 'goal'):
                            # G.add_edge(edge.src.goal, d.goal, label=edge.tactic, color='green')
                            G.add_edge(edge.src.goal, d.goal, label=edge.tactic, color='green')
                            siblings_.append(d.goal)
                            if node.out_edges != None:
                                add_edges(d)
                        else:
                            if type(d) == ErrorNode:
                                pass
                            elif type(d) == ProofFinishedNode:
                                G.add_edge(edge.src.goal, 'proven', color='green', label=edge.tactic)
                    else:
                        if hasattr(edge.src, 'goal') and hasattr(d, 'goal'):
                            G.add_edge(edge.src.goal, d.goal, label=edge.tactic)
                            siblings_.append(d.goal)
                            if node.out_edges != None:
                                add_edges(d)
                        else:
                            if type(d) == ErrorNode:
                                pass
                            elif type(d) == ProofFinishedNode:
                                G.add_edge(edge.src.goal, 'proven', label=edge.tactic)

                if siblings_:
                    siblings.append(siblings_)

    add_edges(trace.tree)

    for i, sib in enumerate(siblings):
        G.add_subgraph(name=f'cluster_{i}')
        subgraph = [sub for sub in G.subgraphs() if not sub.nodes()][0]
        for s in sib:
            subgraph.add_node(s)

    # G.write('test.dot')
    G.draw(filename, prog='dot', format='svg:cairo')
    logger.info(f'Proof tree saved to {filename}')