import glob
import math
import pickle

import dash_cytoscape as cyto
from dash import Dash, html, dcc

from dash import Input, Output, callback
from tqdm import tqdm

from refactor.process_traces import get_traces
from refactor.proof_node import InternalNode, ErrorNode, ProofFinishedNode, Status

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


def render_htps(trace, node_map, rev_map, search_trace):
    siblings = []
    edge_trace, tree, leaves = search_trace

    for data in edge_trace.values():
        edge = data['edge']
        siblings_ = []
        for d in edge.dst:
            if hasattr(edge.src, 'goal') and hasattr(d, 'goal'):
                siblings_.append((node_map[edge.src.goal], node_map[d.goal]))
            elif isinstance(d, ProofFinishedNode):
                siblings_.append((node_map[edge.src.goal], -1))

        if siblings_:
            siblings.append((siblings_, edge))

    processed_nodes = set()

    cluster_nodes = []
    child_nodes = []
    edges = []

    cluster_nodes.append({'data': {'id': 'root', 'label': ''}})

    child_nodes.append(
        {'data': {'id': str(node_map[trace.tree.goal]), 'goal': trace.tree.goal, 'parent': 'root',
                  'label': node_map[trace.tree.goal]}})

    # todo add edges in order of execution
    # todo self loops when cluster repeated?
    for i, (sib, edge) in enumerate(siblings):
        tactic = edge.tactic
        src = [s[0] for s in sib][0]
        dst = [s[1] for s in sib]

        new_nodes = [s for s in dst if s not in processed_nodes]

        if len(dst) == 1 and dst[0] == -1:
            cluster_nodes.append({'data': {'id': str(tactic) + str(src), 'label': '', 'tactic': tactic}})
            child_nodes.append(
                {'data': {'id': 'QED' + str(tactic) + str(src), 'goal': 'QED',
                          'label': 'QED', 'parent': str(tactic) + str(src)}})

            edges.append({'data': {'source': str(src), 'target': str(tactic) + str(src), 'tactic': tactic}, 'classes': 'clusters proven'})
            edges.append({'data': {'source': str(src), 'target': 'QED' + str(tactic) + str(src), 'tactic': tactic}, 'classes': 'hidden'})

        elif new_nodes:
            cluster_nodes.append({'data': {'id': str(src) + str(i) + tactic, 'label': '', 'tactic': tactic}})
            for s in dst:
                if s not in processed_nodes:
                    child_nodes.append(
                        {'data': {'id': str(s), 'goal': trace.nodes[rev_map[s]].goal,
                                  'parent': str(src) + str(i) + tactic,
                                  'label': str(s)}})

                    processed_nodes = processed_nodes | {s}
                    edges.append({'data': {'source': str(src), 'target': s, 'tactic': tactic}, 'classes': 'hidden'})
                else:
                    # If newly seen set of subgoals, with one goal already seen, just connect to source for now
                    edges.append(
                        {'data': {'target': str(s), 'source': str(src) + str(i) + tactic, 'tactic': tactic},
                         'classes': 'nodes'})

            edge_data = edge_trace[(edge.src.goal, tactic)]
            if edge.distance_to_proof() < math.inf:
                edges.append({'data': {'source': str(src), 'target': str(src) + str(i) + tactic, 'tactic': tactic,
                                       'w_score': edge_data['w_score'], 'visit_count': edge_data['visit_count']},
                              'classes': 'clusters proven'})
            else:
                edges.append({'data': {'source': str(src), 'target': str(src) + str(i) + tactic, 'tactic': tactic,
                                       'w_score': edge_data['w_score'], 'visit_count': edge_data['visit_count']},
                              'classes': 'clusters'})

        else:
            edge_data = edge_trace[(edge.src.goal, tactic)]
            for s in dst:
                edges.append({'data': {'source': str(src), 'target': str(s), 'tactic': tactic,
                                       'w_score': edge_data['w_score'], 'visit_count': edge_data['visit_count']},
                              'classes': 'nodes'})

    leaf_goals = [l[0].goal for l in leaves]

    expanded_nodes = set([g for (g,t), v in edge_trace.items()])

    for node in child_nodes:
        classes = ""
        goal = node['data']['goal']
        if goal != 'QED':
            # if trace.nodes[goal].status == Status.PROVED:
            #     classes += " proven"
            if node['data']['goal'] in tree:
                classes += " tree"
            if goal in leaf_goals:
                classes += " leaves"
            if trace.nodes[goal].status == Status.FAILED:
                classes += " expanded"
        else:
            classes += " proven"
        node['classes'] = classes

    elements = cluster_nodes + child_nodes + edges
    return elements


def render_full_trace(trace):
    siblings = []

    node_map = {}
    i = 0
    for edge in trace.tac_trace:
        if edge.src.goal not in node_map:
            node_map[edge.src.goal] = i
            i += 1
            # todo remove this (old traces were wrong)
            if edge.src.goal not in trace.nodes:
                trace.nodes[edge.src.goal] = edge.src
        for d in edge.dst:
            if isinstance(d, InternalNode):
                if d.goal not in node_map:
                    node_map[d.goal] = i
                    i += 1
                    if d.goal not in trace.nodes:
                        trace.nodes[d.goal] = d

    rev_map = {v: k for k, v in node_map.items()}

    def add_edges(node):
        if not node.out_edges:
            return
        for j, edge in enumerate(node.out_edges):
            siblings_ = []
            for i, d in enumerate(edge.dst):
                if hasattr(edge.src, 'goal') and hasattr(d, 'goal'):
                    siblings_.append((node_map[edge.src.goal], node_map[d.goal]))

                    if node.out_edges != None:
                        add_edges(d)
                else:
                    if type(d) == ErrorNode:
                        pass
                    elif type(d) == ProofFinishedNode:
                        pass
            if siblings_:
                siblings.append((siblings_, edge))

    add_edges(trace.tree)

    processed_nodes = set()

    cluster_nodes = []
    child_nodes = []
    edges = []

    cluster_nodes.append({'data': {'id': 'root', 'label': ''}})

    child_nodes.append(
        {'data': {'id': str(node_map[trace.tree.goal]), 'goal': trace.tree.goal, 'parent': 'root',
                  'label': node_map[trace.tree.goal]}})

    # todo add edges in order of execution
    # todo self loops when cluster repeated?
    for i, (sib, edge) in enumerate(siblings):
        tactic = edge.tactic
        src = [s[0] for s in sib][0]
        dst = [s[1] for s in sib]

        new_nodes = [s for s in dst if s not in processed_nodes]

        if new_nodes:
            cluster_nodes.append({'data': {'id': str(src) + str(i) + tactic, 'label': '', 'tactic': tactic}})
            for s in dst:
                if s not in processed_nodes:
                    child_nodes.append(
                        {'data': {'id': str(s), 'goal': trace.nodes[rev_map[s]].goal,
                                  'parent': str(src) + str(i) + tactic,
                                  'label': str(s)}})

                    processed_nodes = processed_nodes | {s}
                    edges.append({'data': {'source': str(src), 'target': s, 'tactic': tactic}, 'classes': 'hidden'})
                else:
                    # If newly seen set of subgoals, with one goal already seen, just connect to source for now
                    edges.append(
                        {'data': {'target': str(s), 'source': str(src) + str(i) + tactic, 'tactic': tactic},
                         'classes': 'nodes'})

            if edge.distance_to_proof() < math.inf:
                edges.append({'data': {'source': str(src), 'target': str(src) + str(i) + tactic, 'tactic': tactic},
                              'classes': 'clusters proven'})
            else:
                edges.append({'data': {'source': str(src), 'target': str(src) + str(i) + tactic, 'tactic': tactic},
                              'classes': 'clusters'})

    for node in child_nodes:
        if trace.nodes[node['data']['goal']].status == Status.PROVED:
            node['classes'] = "proven"

    elements = cluster_nodes + child_nodes + edges
    return elements


if __name__ == '__main__':
    # traces = get_traces('../experiments/runs/leandojo/sample_bestfs_2023_11_29/12_14_36/traces/set.is_wf_min_singleton')
    # traces = get_traces('../experiments/runs/leandojo/sample_bestfs_2023_11_29/14_16_34/traces/is_lub.exists_between_sub_self')

    # traces = get_traces('../experiments/runs/leandojo/sample_bestfs_2023_11_29/14_30_44/traces/algebraic_geometry.Scheme.pullback.p_comm')
    traces = get_traces('../experiments/runs/leandojo/sample_bestfs_2023_11_29/15_01_36/traces/measure_theory.norm_integral_le_of_norm_le')


    cyto.load_extra_layouts()


    # elements = render_full_trace(traces[0])

    trace = traces[0]

    node_map = {}
    i = 0
    for edge in trace.trace:
        if edge.src.goal not in node_map:
            node_map[edge.src.goal] = i
            i += 1
        for d in edge.dst:
            if isinstance(d, InternalNode):
                if d.goal not in node_map:
                    node_map[d.goal] = i
                    i += 1
                    if d.goal not in trace.nodes:
                        trace.nodes[d.goal] = d

    rev_map = {v: k for k, v in node_map.items()}

    search_trace = trace.data['search_trace']

    elements = render_htps(node_map=node_map,
                           search_trace=search_trace[1],
                           rev_map=rev_map,
                           trace=trace)

    app = Dash(__name__)

    app.layout = html.Div([
        cyto.Cytoscape(
            id='cytoscape-compound',
            layout={'name': 'dagre'},
            style={'width': '100%', 'height': '900px'},
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {'content': 'data(label)',
                              'text-valign': 'center'}
                },
                {
                    'selector': '.nodes',
                    'style': {'line-style': 'solid',
                              'color': 'yellow',
                              # 'target-arrow-color': '#000',
                              # 'target-arrow-shape': 'triangle',
                              'width': '0.5'
                              }
                },
                {
                    'selector': '.clusters',
                    'style': {'line-style': 'dashed',
                              'target-arrow-color': '#000)',
                              'target-arrow-shape': 'triangle',
                              }
                },
                {
                    'selector': '.hidden',
                    'style': {'line-style': 'dashed', 'width': '0'}
                },
                {
                    'selector': '.dupes',
                    'style': {'line-style': 'dashed'}

                },
                {
                    'selector': '.expanded',
                    'style': {'background-color': 'red',
                              'line-color': 'red'}

                },
                {
                    'selector': '.proven',
                    'style': {'background-color': 'green',
                              'line-color': 'green'}

                },
                {
                    'selector': '.tree',
                    'style': {'background-color': 'yellow',
                              'line-color': 'yellow'}

                },
                {
                    'selector': '.leaves',
                    'style': {'background-color': 'blue',
                              'line-color': 'blue'}

                },
            ],
            elements=elements
        ),
        dcc.Markdown(id='cytoscape-tapNodeData-json', style={'white-space': 'pre'}),
        dcc.Markdown(id='cytoscape-tapEdgeData-json'),
        dcc.Dropdown(
            id='dropdown-update-layout',
            value=0,
            clearable=False,
            options=[
                {'label': str(index), 'value': index}
                for index in range(len(trace.data['search_trace']))
            ]
        ),
    ])


    @callback(Output('cytoscape-compound', 'elements'),
              Input('dropdown-update-layout', 'value'))
    def update_layout(value):
        return render_htps(node_map=node_map,
                           search_trace=search_trace[value],
                           rev_map=rev_map,
                           trace=trace)


    # todo add scores
    @callback(Output('cytoscape-tapNodeData-json', 'children'),
              Input('cytoscape-compound', 'tapNodeData'))
    def displayTapNodeData(data):
        if data and 'goal' in data:
            ret = data['goal'] + '\n'
            return 'Goal:\n' + ret
        elif data and 'tactic' in data:
            return 'From tactic: \n' + data['tactic']


    @callback(Output('cytoscape-tapEdgeData-json', 'children'),
              Input('cytoscape-compound', 'tapEdgeData'))
    def displayTapEdgeData(data):
        if data:
            if 'w_score' in data:
                return 'Tactic: \n' + data['tactic'] + '\nW score: ' + str(data['w_score']) + '\nVisit Count:' + str(data['visit_count'])
            else:
                return 'Tactic: \n' + data['tactic']


    app.run(debug=True)
