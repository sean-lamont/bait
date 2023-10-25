# '''''
#
# Skeleton code for modules needed to abstractly define end-to-end experiment
#
# '''''
#
#
#
#
# '''
# Abstract proof state containing all relevant information for search.
# Should include:
# - goal/state (either single goal or a full state)
# - status (proven, failed, open)
# - ignore/is_explored
# - distance to proof
# - visit count
# - out_edges
# - in_edges
# - context
# - ancestors
# - optional provable/up score
# - children (for computing total visit count)
# - depth
# - methods for extracting proof, updating context, status, ancestors,
# '''
# @dataclass
# class ProofNode:
#
#
# '''
# Edges should be more or less the same as they are currently defined:
# '''
# @dataclass
# class Edge:
#     tactic: str
#     src: InternalNode = field(repr=False)
#     dst: List[Node] = field(repr=False)
#     logprob: float
#     time: float
#
#     def distance_to_proof(self) -> float:
#         return 1 + sum([d.distance_to_proof for d in self.dst])
#
#
#
#
# '''
# Search algorithms based on abstract node as above
# '''
#
#
# def best_fs():
#     goal = queue.pop()
#     tactics = get_tacs(goal)
#     responses = [env.step(goal, tactics)]
#     process_response(response)
#
# def bfs():
#     queue = [root]
#     while root != Proved or not timeout or not queue:
#         tactics = get_tacs(goal)
#         responses = [env.step(goal, tactics)]
#         process_response(response)
#         queue.push(responses)
#
# def HTPS():
#     # run puct algo as per paper
#     leaves = get_hypertree(state)
#     tactics = get_tacs(leaves)
#     responses = [env.step(leaves, tactics)]
#     process_response(responses)
#
# def updown():
#     goal = argmax(goal.provable_score * goal.context)
#     tactics = get_tacs(goal)
#     responses = [env.step(goal, tactics)]
#     process_response(responses)
#
#
# def process_response(response):
#     # should include goal, tactic, logprobs, enough to reconstruct the state when running sequentially
#     add_to_database(response)
#
#     # add all children, update contexts, visits, scores, status etc.
#     update_state(response)
#
#
# def get_tacs():
#     # get allowed premises or retrieved context
#     # will be either generative model given state,
#     # or will be tactic from fixed set followed by premise selection
#
#
# '''
# Generic wrapper on search
# '''
#
# def run():
#     while root.status != status.PROVED or not timeout:
#         trace = search()
#
#     process_trace(trace)
#
#
# def process_trace(trace):
#     # save/process trace to database/storage
#
#     # update_params based on trace if online policy