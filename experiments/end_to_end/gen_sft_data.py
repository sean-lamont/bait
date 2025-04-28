import glob
from google import genai
from google.genai import types
from experiments.end_to_end.proof_node import Status
from experiments.end_to_end.proof_node import ErrorNode, ProofFinishedNode
import pickle

import seaborn as sns
from tqdm import tqdm

paths = {
    'bfs_2': "../runs/internlm/internlm/2025_03_13/16_14_49/traces/0/",
}


def get_proven_traces(path):
    files = glob.glob(path + '*', recursive=True)
    # traces_ = []

    proven = []
    for file in tqdm(files):
        try:
            # traces_.append(pickle.load(open(file, "rb")))
            trace = pickle.load(open(file, "rb"))
            if trace.proof:
                proven.append(trace)
        except:
            # print ('Failed to load:', file)
            continue

    return proven


proven_traces = get_proven_traces(paths['bfs_2'])

lens = [len(trace.proof) for trace in proven_traces]
sns.histplot(lens, kde=True)


def get_deepest_children(root):
    # go through tree and map each node to the depth of deepest child

    # get nodes in order via BFS
    to_search = [root]
    nodes = [root]
    visited = set()
    node_depths = {}
    while to_search:
        cur_node = to_search.pop(0)
        if cur_node in visited:
            continue
        visited.add(cur_node)

        if cur_node.out_edges:
            # assumes no subgoal separation
            to_search.extend([e.dst[0] for e in cur_node.out_edges if
                              not isinstance(e.dst[0], ErrorNode) and not isinstance(e.dst[0], ProofFinishedNode) and
                              e.dst[0] not in visited])

            nodes.extend([e.dst[0] for e in cur_node.out_edges if
                          not isinstance(e.dst[0], ErrorNode) and not isinstance(e.dst[0], ProofFinishedNode) and e.dst[
                              0] not in visited])

    # now we have all nodes, go through and get max depth (all children will be filled in as in BFS order)
    for node in nodes[::-1]:
        if not node.out_edges:
            node_depths[node] = 0
        else:
            node_depths[node] = 1 + max([node_depths[e.dst[0]] for e in node.out_edges if
                                         not isinstance(e.dst[0], ErrorNode) and not isinstance(e.dst[0],
                                                                                                ProofFinishedNode) and
                                         e.dst[0] in node_depths] + [0])

    return node_depths


def get_sft_data(trace):
    proof = trace.proof
    idx = 0

    error_data = []
    success_data = []
    backtrack_data = []

    cur_node = trace.tree
    node_depths = get_deepest_children(cur_node)

    prev_goal = ''
    while idx < len(proof):
        cur_tactic = proof[idx]

        success_data.append({
            'cur_goal': cur_node.goal,
            'tactic': cur_tactic,
            'proof_so_far': proof[:idx],
            'original_goal': trace.theorem.full_name,
            'prev_goal': prev_goal,
        })
        # check if there is an error coming from the current node
        if cur_node.out_edges:
            # check if any of the edges are error nodes
            for edge in cur_node.out_edges:
                if isinstance(edge.dst[0], ErrorNode):
                    # check if the error node has the same goal as the current node
                    error_data.append({
                        'cur_goal': cur_node.goal,
                        'tactic': cur_tactic,
                        'proof_so_far': proof[:idx],
                        'original_goal': trace.theorem.full_name,
                        'error_tactic': edge.tactic,
                        'error_message': edge.dst[0].inner.message,
                        'prev_goal': prev_goal,
                    })
                    # just take one for now
                    break

        # go to next node
        next_node = [a.dst[0] for a in cur_node.out_edges if a.tactic == cur_tactic][0]

        # check for backtrack data
        if cur_node.out_edges:
            for edge in cur_node.out_edges:
                if not isinstance(edge.dst[0], ErrorNode) and edge.tactic != cur_tactic and edge.dst[
                    0].status != Status.PROVED and edge.dst[0] in node_depths:
                    # for nodes with a direct proof, any open should be backtracked
                    if isinstance(next_node, ProofFinishedNode):
                        backtrack_data.append({
                            'cur_goal': cur_node.goal,
                            'tactic': cur_tactic,
                            'proof_so_far': proof[:idx],
                            'original_goal': trace.theorem.full_name,
                            'backtrack_tactic': edge.tactic,
                            'backtrack_goal': edge.dst[0].goal,
                            'prev_goal': prev_goal,
                        })
                        break


                    # otherwise, check if sibling has been explored at least as much as proven node path
                    elif node_depths[edge.dst[0]] >= node_depths[next_node]:
                        backtrack_data.append({
                            'cur_goal': cur_node.goal,
                            'tactic': cur_tactic,
                            'proof_so_far': proof[:idx],
                            'original_goal': trace.theorem.full_name,
                            'backtrack_tactic': edge.tactic,
                            'backtrack_goal': edge.dst[0].goal,
                            'prev_goal': prev_goal,
                        })

                        break

        # go to next node
        prev_goal = cur_node.goal
        cur_node = next_node
        idx += 1

    return success_data, error_data, backtrack_data


test_data = []

for t in proven_traces:
    test_data.append(get_sft_data(t))

prompt = ('You are an expert in Lean 4 theorem proving. You are given '
          'some information about a proof search attempt, including the original goal name [ORIGINAL_GOAL],'
          'the current goal state [CUR_GOAL], the previous goal state [PREV_GOAL], the proof so far from the root node [PROOF_SO_FAR], previous tactics which may have failed [FAILED_TACTICS], including their error message [ERROR_MESSAGE]. Using this information, provide a rationale as to why the chosen tactic [TACTIC] was used, assuming it was chosen based only on the information above. If the tactic is [BACKTRACK], then this means that the last tactic was successful in Lean, but no useful progress was made in the proof, with the best tactic being [BACKTRACK] to the previous goal. First explain the goal,'
          'then explain whether any progress has been made compared to the previous goal if it exists. Then '
          'explain previously failed tactics, and why they failed. Finally, explain the plan for the proof and the rationale for choosing the given tactic. If the tactic is [BACKTRACK], only provide a rationale for why you are backtracking. You only need to return the rationale, not the entire proof state as provided.')

proof_data = []
error_data = []
backtrack_data = []
for val in test_data:
    llm_input = ''
    for item in val[0]:
        llm_input = llm_input + ('[ORIGINAL_GOAL]\n' + item['original_goal'] + '\n')
        llm_input = llm_input + ('[CUR_GOAL]\n' + item['cur_goal'] + '\n')
        llm_input = llm_input + ('[PREV_GOAL]\n' + item['prev_goal'] + '\n')
        llm_input = llm_input + ('[FAILED_TACTICS]\n' + '' + '\n')
        llm_input = llm_input + ('[ERROR_MESSAGE]\n' + '' + '\n')
        llm_input = llm_input + ('[PROOF_SO_FAR]\n' + str(item['proof_so_far']) + '\n')
        llm_input = llm_input + ('[TACTIC]\n' + item['tactic'] + '\n')
    proof_data.append(llm_input)
    llm_input = ''
    for item in val[1]:
        llm_input = llm_input + ('[ORIGINAL_GOAL]\n' + item['original_goal'] + '\n')
        llm_input = llm_input + ('[CUR_GOAL]\n' + item['cur_goal'] + '\n')
        llm_input = llm_input + ('[PREV_GOAL]\n' + item['prev_goal'] + '\n')
        llm_input = llm_input + ('[PROOF_SO_FAR]\n' + str(item['proof_so_far']) + '\n')
        llm_input = llm_input + ('[FAILED_TACTICS]\n' + str(item['error_tactic']) + '\n')
        llm_input = llm_input + ('[ERROR_MESSAGE]\n' + str(item['error_message']) + '\n')
        llm_input = llm_input + ('[TACTIC]\n' + item['tactic'] + '\n')
    error_data.append(llm_input)
    llm_input = ''
    for item in val[2]:
        # generate two examples: one for backtracking, one for taking correct path given backtrack
        llm_input = llm_input + ('[ORIGINAL_GOAL]\n' + item['original_goal'] + '\n')
        llm_input = llm_input + ('[CUR_GOAL]\n' + item['backtrack_goal'] + '\n')
        llm_input = llm_input + ('[PREV_GOAL]\n' + item['cur_goal'] + '\n')
        item['proof_so_far'].append(item['backtrack_tactic'])
        llm_input = llm_input + ('[PROOF_SO_FAR]\n' + str(item['proof_so_far']) + '\n')
        llm_input = llm_input + ('[FAILED_TACTICS]\n' + '' + '\n')
        llm_input = llm_input + ('[ERROR_MESSAGE]\n' + '\n')
        llm_input = llm_input + ('[TACTIC]\n' + '[BACKTRACK]' + '\n')
    backtrack_data.append(llm_input)

annotated_backtrack_data = []
annotated_proof_data = []
annotated_error_data = []


def generate(input):
    client = genai.Client(
        api_key='AIzaSyAILGrHm2DT7nA859hvmjDwj5fEL4cL2W8',
    )

    model = "gemini-2.5-flash-preview-04-17"
    # model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt + input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    out = ""
    for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
    ):
        # print(chunk.text, end="")
        try:
            out += chunk.text
        except:
            continue

    return out


for input in tqdm(backtrack_data):
    annotated_backtrack_data.append((input, generate(input)))

for input in tqdm(proof_data):
    annotated_proof_data.append((input, generate(input)))

for input in tqdm(error_data):
    annotated_error_data.append((input, generate(input)))

test_ = annotated_backtrack_data

for input, response in annotated_backtrack_data:
    print(input)
    print(response)
    print('---')

for input, response in annotated_error_data:
    print(input)
    print(response)
    print('---')

for input, response in annotated_proof_data:
    print(input)
    print(response)
    print('---')
