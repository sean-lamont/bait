from refactor.proof_node import Status, ErrorNode, ProofFinishedNode, InternalNode
import pickle
import pygraphviz as pgv
from loguru import logger


def render_proof(path, filename=None):
    with open(path, 'rb') as f:
        trace = pickle.load(f)

    logger.info(f'Rendering proof of {trace.tree.goal}..')

    if not filename:
        filename = 'figures/' + trace.theorem + '.svg'

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


def render_full_trace(trace, filename=None):
    # with open(path, 'rb') as f:
    #     trace = pickle.load(f)

    logger.info(f'Rendering proof of {trace.tree.goal}..')

    if not filename:
        filename = 'figures/' + trace.theorem + '.svg'

    siblings = []

    G = pgv.AGraph(name='root', compound=True)

    node_map = {}
    i = 0
    for edge in trace.tac_trace:
        if edge.src.goal not in node_map:
            node_map[edge.src.goal] = i
            i += 1
        for d in edge.dst:
            if isinstance(d, InternalNode):
                if d.goal not in node_map:
                    node_map[d.goal] = i
                    i += 1

    # node_map = {goal: i for i, goal in enumerate(trace.nodes.keys())}

    rev_map = {v: k for k, v in node_map.items()}

    def add_edges(node):
        if not node.out_edges:
            return
        for j, edge in enumerate(node.out_edges):
            siblings_ = []
            for i, d in enumerate(edge.dst):
                if hasattr(edge.src, 'goal') and hasattr(d, 'goal'):
                    # G.add_edge(edge.src.goal, d.goal, label=edge.tactic, color='green')
                    # G.add_edge(edge.src.goal, d.goal, label=edge.tactic, color='green')
                    # if all([d_.status == Status.PROVED for d_ in edge.dst]):
                    #     G.add_edge(node_map[edge.src.goal], node_map[d.goal],
                    #                label=(j, i), color='green')
                    # else:
                    #     G.add_edge(node_map[edge.src.goal], node_map[d.goal], label=(j, i))

                    siblings_.append((node_map[edge.src.goal], node_map[d.goal]))

                    if node.out_edges != None:
                        add_edges(d)
                else:
                    if type(d) == ErrorNode:
                        pass
                    elif type(d) == ProofFinishedNode:
                        pass
            if siblings_:
                siblings.append(siblings_)

    add_edges(trace.tree)

    processed_nodes = set()
    for i, sib in enumerate(siblings):
        src = [s[0] for s in sib][0]
        dst = [s[1] for s in sib]

        new_nodes = [s for s in dst if s not in processed_nodes]

        if new_nodes:
            G.add_subgraph(name=f'cluster_{i}')
            # get most recent subgraph (i.e. with no nodes)
            subgraph = [sub for sub in G.subgraphs() if not sub.nodes()][0]

            for s in dst:
                if s not in processed_nodes:
                    if trace.nodes[rev_map[s]].status == Status.FAILED:
                        print ('asdf')
                        subgraph.add_node(s, color='red')

                    elif trace.nodes[rev_map[s]].status == Status.PROVED:
                        subgraph.add_node(s, color='green')
                    else:
                        subgraph.add_node(s)
                    processed_nodes = processed_nodes | {s}
                else:
                    # If newly seen set of subgoals, with one goal already seen, just connect to source for now
                    G.add_edge(new_nodes[0], s, color='yellow')

            G.add_edge(src, new_nodes[0], lhead=f'cluster_{i}')

    # G.write('test.dot')
    G.draw(filename, prog='dot', format='svg:cairo')
    logger.info(f'Proof tree saved to {filename}')