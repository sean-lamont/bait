import math

import dash_cytoscape as cyto
from dash import Dash, html, dcc
from dash import Input, Output, callback

from experiments.end_to_end.process_traces import get_traces
from experiments.end_to_end.proof_node import InternalNode, ProofFinishedNode, Status

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# todo
# def render_bestfs():

def render_updown(trace, node_map, rev_map, search_trace):
    siblings = []

    (fringe_goals, node_scores, initial_scores, updated_scores), responses = search_trace

    max_ind = -1
    for response in responses:
        ind = trace.trace.index(response)
        if ind > max_ind:
            max_ind = ind

    responses = trace.trace[:max_ind + 1]

    for edge in responses:
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
                  'label': node_map[trace.tree.goal], }})

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

            edges.append({'data': {'source': str(src), 'target': str(tactic) + str(src), 'tactic': tactic},
                          'classes': 'clusters proven'})
            edges.append({'data': {'source': str(src), 'target': 'QED' + str(tactic) + str(src), 'tactic': tactic},
                          'classes': 'hidden'})

        elif new_nodes:
            cluster_nodes.append({'data': {'id': str(src) + str(i) + tactic, 'label': '', 'tactic': tactic}})
            for s in dst:
                if s not in processed_nodes:
                    goal = trace.nodes[rev_map[s]].goal
                    data = {'id': str(s), 'goal': goal,
                            'parent': str(src) + str(i) + tactic,
                            'label': str(s),
                            }
                    if goal in node_scores:
                        data['final_score'] = math.exp(node_scores[goal])
                    else:
                        data['final_score'] = 'Explored'
                    if goal in initial_scores:
                        data['initial_score'] = math.exp(initial_scores[goal])
                    if goal in updated_scores:
                        data['updated_score'] = math.exp(updated_scores[goal])

                    child_nodes.append({'data': data})

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

        # else:
        # for s in dst:
        #     edges.append({'data': {'source': str(src), 'target': str(s), 'tactic': tactic},
        #                   'classes': 'nodes'})

    for node in child_nodes:
        classes = ""
        goal = node['data']['goal']
        if goal != 'QED':
            if goal in fringe_goals:
                classes += " leaves"
            if trace.nodes[goal].visit_count == 64:
                classes += " expanded"
        else:
            classes += " proven"
        node['classes'] = classes

    elements = cluster_nodes + child_nodes + edges
    return elements


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

            edges.append({'data': {'source': str(src), 'target': str(tactic) + str(src), 'tactic': tactic},
                          'classes': 'clusters proven'})
            edges.append({'data': {'source': str(src), 'target': 'QED' + str(tactic) + str(src), 'tactic': tactic},
                          'classes': 'hidden'})

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
    #
    expanded_nodes = set([g for (g, t), v in edge_trace.items()])

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


if __name__ == '__main__':

    # traces = get_traces(
    #     "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/set.definable.compl")

    # good example of updown not exploring, only one edge from the root node is explored,
    # as the others are initially estimated very low:
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/tsub_lt_tsub_iff_left_of_le")

    # as above
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/X_in_terms_of_W_vars_subset")

    # nodes 6,7,9 are all semantically identical, varying only with renaming of a variable, yet all have
    # very different scores. Indicates the goal model doesn't have a good understanding (it should learn
    # that renamed variables don't impact the provability)
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/upper_set.coe_Inf")

    # another example of same hypotheses (although scored similarly here)
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/subgroup.subset_closure")

    # Failing after expanding all valid nodes, note that path is terminated once member of context is found to fail
    # Think HTPS wouldn't pick up on this? (node 6,7,8,9)
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/simple_graph.nonempty_of_pos_dist")

    # very large graph, with timeout. Again, lot's of variable renaming with same goal
    # "../experiments/runs/leandojo/sample_bestfs_2023_11_29/20_30_17/traces/set.definable.compl")

    # holist test:
    traces = get_traces('../experiments/runs/end_to_end_holist/test_2024_01_08/15_46_13/traces/*')

    cyto.load_extra_layouts()

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

    # search_trace = trace.data['search_trace']

    elements = render_bestfs(node_map=node_map,
                             rev_map=rev_map,
                             trace=trace,
                             step=0)

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
                    'selector': '.tree',
                    'style': {'background-color': 'yellow',
                              'line-color': 'yellow'}

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
                # for index in range(len(trace.data['search_trace']))
                for index in range(len(trace.trace))
            ]
        ),
    ])


    @callback(Output('cytoscape-compound', 'elements'),
              Input('dropdown-update-layout', 'value'))
    def update_layout(value):
        # return render_htps(node_map=node_map,
        #                    search_trace=search_trace[value],
        #                    rev_map=rev_map,
        #                    trace=trace)

        # return render_updown(node_map=node_map,
        #                      search_trace=search_trace[value],
        #                      rev_map=rev_map,
        #                      trace=trace)

        return render_bestfs(node_map=node_map,
                             # search_trace=search_trace[value],
                             rev_map=rev_map,
                             trace=trace,
                             step=value)


    # todo add scores
    @callback(Output('cytoscape-tapNodeData-json', 'children'),
              Input('cytoscape-compound', 'tapNodeData'))
    def displayTapNodeData(data):
        if data and 'goal' in data:
            ret = data['goal'] + '\n'
            if 'initial_score' in data:
                ret += f'Initial score:  {data["initial_score"]} \n' \
                       + f"Updated score: {data['updated_score']}\n" \
                       + f"Final score: {data['final_score']}\n"

            return 'Goal:\n' + ret
        elif data and 'tactic' in data:
            return 'From tactic: \n' + data['tactic']


    @callback(Output('cytoscape-tapEdgeData-json', 'children'),
              Input('cytoscape-compound', 'tapEdgeData'))
    def displayTapEdgeData(data):
        if data:
            if 'w_score' in data:
                return 'Tactic: \n' + data['tactic'] + '\nW score: ' + str(data['w_score']) + '\nVisit Count:' + str(
                    data['visit_count'])
            else:
                return 'Tactic: \n' + data['tactic']


    app.run(debug=True)