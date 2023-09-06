import warnings

import einops
import torch.nn.functional as F
import torch.optim
from torch.distributions import Categorical

from experiments.hol4_tactic_zero.rl.old.rl_data_module import *

warnings.filterwarnings('ignore')

MORE_TACTICS = True
if not MORE_TACTICS:
    thms_tactic = ["simp", "fs", "metis_tac"]
    thm_tactic = ["irule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac"]
else:
    thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    thm_tactic = ["irule", "drule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac", "EQ_TAC"]

tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic

def get_tac(tac_input, tac_net, device):
    tac_probs = tac_net(tac_input)
    tac_m = Categorical(tac_probs)
    tac = tac_m.sample()
    tac_prob = tac_m.log_prob(tac)
    tac_tensor = tac.to(device)
    return tac_tensor, tac_prob

def select_goal_fringe(history,
                       encoder_goal,
                       graph_db,
                       token_enc,
                       context_net,
                       device,
                       data_type='graph',
                       replay_fringe=None):

    representations, context_set, fringe_sizes = gather_encoded_content_gnn(history, encoder_goal,
                                                                            device, graph_db=graph_db,
                                                                            token_enc=token_enc, data_type=data_type)
    context_scores = context_net(representations)
    contexts_by_fringe, scores_by_fringe = split_by_fringe(context_set, context_scores, fringe_sizes)
    fringe_scores = []

    for s in scores_by_fringe:
        fringe_score = torch.sum(s)
        fringe_scores.append(fringe_score)

    fringe_scores = torch.stack(fringe_scores)
    fringe_probs = F.softmax(fringe_scores, dim=0)
    fringe_m = Categorical(fringe_probs)

    if replay_fringe is not None:
        fringe = replay_fringe
    else:
        fringe = fringe_m.sample()

    fringe_prob = fringe_m.log_prob(fringe)
    # take the first context in the chosen fringe
    target_context = contexts_by_fringe[fringe][0]
    target_goal = target_context["polished"]["goal"]
    target_representation = representations[context_set.index(target_context)]

    return target_representation, target_goal, fringe, fringe_prob

def get_term_tac(target_goal, target_representation, tac, term_net, induct_net, device, token_enc, replay_term=None):
    target_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc)

    arg_probs = []
    tokens = [[t] for t in target_graph.labels if t[0] == "V"]
    token_inds = [i for i, t in enumerate(target_graph.labels) if t[0] == "V"]

    if tokens:
        # pass whole graph through Induct GNN
        induct_graph = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(target_goal), token_enc).to(
            device)
        induct_graph.edge_attr = induct_graph.edge_attr.long()
        induct_nodes = induct_net(induct_graph)

        # select representations of Variable nodes with ('V' label only)
        token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(device))

        # pass through term_net as before
        target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))
        candidates = torch.cat([token_representations, target_representations], dim=1)
        scores = term_net(candidates, tac)
        term_probs = F.softmax(scores, dim=0)
        term_m = Categorical(term_probs.squeeze(1))

        if replay_term is None:
            term = term_m.sample()
        else:
            term = torch.tensor([tokens.index(["V" + replay_term])]).to(device)

        arg_probs.append(term_m.log_prob(term))
        tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
        tactic = "Induct_on `{}`".format(tm)

    else:
        arg_probs.append(torch.tensor(0))
        tactic = "Induct_on"

    return tactic, arg_probs

def get_arg_tac(target_representation,
                num_args,
                encoded_fact_pool,
                tac,
                candidate_args,
                env,
                device,
                arg_net,
                arg_len,
                reverse_database,
                replay_arg=None):

    hidden0 = hidden1 = target_representation
    hidden0 = hidden0.to(device)
    hidden1 = hidden1.to(device)

    hidden = (hidden0, hidden1)
    # concatenate the candidates with hidden states.

    hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
    hiddenl = [hc.unsqueeze(0) for _ in range(num_args)]
    hiddenl = torch.cat(hiddenl)

    candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
    candidates = candidates.to(device)
    input = tac

    # run it once before predicting the first argument
    hidden, _ = arg_net(input, candidates, hidden)

    # the indices of chosen args
    arg_step = []
    arg_step_probs = []

    if tactic_pool[tac] in thm_tactic:
        arg_len = 1
    else:
        # arg_len = config['arg_len']  # ARG_LEN
        arg_len = arg_len

    for i in range(arg_len):
        hidden, scores = arg_net(input, candidates, hidden)
        arg_probs = F.softmax(scores, dim=0)
        arg_m = Categorical(arg_probs.squeeze(1))

        if replay_arg is None:
            arg = arg_m.sample()
        else:
            if isinstance(replay_arg, list):
                try:
                    name_parser = replay_arg[i].split(".")
                except:
                    print(i)
                    print(replay_arg)
                    exit()
                theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                theorem_name = name_parser[1]
                # todo not sure if reverse_database will work...
                true_arg_exp = reverse_database[(theory_name, theorem_name)]
            else:
                name_parser = replay_arg.split(".")
                theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                theorem_name = name_parser[1]
                true_arg_exp = reverse_database[(theory_name, theorem_name)]

            arg = torch.tensor(candidate_args.index(true_arg_exp)).to(device)

        arg_step.append(arg)
        arg_step_probs.append(arg_m.log_prob(arg))

        hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
        hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

        # encoded chosen argument
        input = encoded_fact_pool[arg].unsqueeze(0)

        # renew candidates
        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
        hiddenl = [hc.unsqueeze(0) for _ in range(num_args)]
        hiddenl = torch.cat(hiddenl)

        # appends both hidden and cell states (when paper only does hidden?)
        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
        candidates = candidates.to(device)

    tac = tactic_pool[tac]
    arg = [candidate_args[j] for j in arg_step]

    tactic = env.assemble_tactic(tac, arg)
    return tactic, arg_step_probs

def get_replay_tac(true_tactic_text):
    if true_tactic_text in no_arg_tactic:
        true_tac_text = true_tactic_text
        true_args_text = None
    else:
        tac_args = re.findall(r'(.*?)\[(.*?)\]', true_tactic_text)
        tac_term = re.findall(r'(.*?) `(.*?)`', true_tactic_text)
        tac_arg = re.findall(r'(.*?) (.*)', true_tactic_text)
        if tac_args:
            true_tac_text = tac_args[0][0]
            true_args_text = tac_args[0][1].split(", ")
        elif tac_term:  # order matters # TODO: make it irrelavant
            true_tac_text = tac_term[0][0]
            true_args_text = tac_term[0][1]
        elif tac_arg:  # order matters because tac_arg could match () ``
            true_tac_text = tac_arg[0][0]
            true_args_text = tac_arg[0][1]
        else:
            true_tac_text = true_tactic_text
    return true_tac_text, true_args_text
