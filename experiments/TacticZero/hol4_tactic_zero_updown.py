import traceback

import einops
import torch.nn.functional as F
import torch.optim
from torch.distributions import Categorical
from torch_geometric.data import Batch

from data.HOL4.utils import ast_def
from environments.HOL4.tacticzero_old.updown_env import *
from experiments.TacticZero.tactic_zero_module import TacticZeroLoop

'''

HOL4 Implementation of TacticZero as outlined in orignial paper with graph based environment state

'''


class HOL4TacticZero(TacticZeroLoop):
    def __init__(self,
                 config,
                 goal_net,
                 tac_net,
                 arg_net,
                 term_net,
                 induct_net,
                 encoder_premise,
                 encoder_goal,
                 tactics,
                 converter,
                 proof_db
                 ):

        super().__init__(config)

        self.goal_net = goal_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal

        self.thms_tactic = list(tactics.thms_tactic)
        self.thm_tactic = list(tactics.thm_tactic)
        self.term_tactic = list(tactics.term_tactic)
        self.no_arg_tactic = list(tactics.no_arg_tactic)
        self.tactic_pool = list(tactics.tactic_pool)
        self.converter = converter
        self.proof_db = proof_db
        self.proof_logs = []

        self.config = config
        self.dir = self.config.exp_config.directory
        self.replay_dir = self.dir + '/replays.pt'
        os.makedirs(self.dir, exist_ok=True)

        self.setup_replays()

    def setup_replays(self):
        if os.path.exists(self.dir + '/replays.pt'):
            self.replays = torch.load(self.replay_dir)
            self.cumulative_proven = list(self.replays.keys())
        else:
            logging.debug(f"Creating new replay dir {self.replay_dir}")
            self.replays = {}

    def get_tac(self, tac_input):
        tac_probs = self.tac_net(tac_input)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        tac_prob = tac_m.log_prob(tac)
        tac_tensor = tac.to(self.device)
        return tac_tensor, tac_prob

    # determine term for induction based on data type (graph, fixed, sequence)
    def get_term_tac(self, target_goal, target_representation, tac, replay_term=None):
        arg_probs = []

        induct_expr = self.converter([target_goal])

        if self.config.data_config.type == 'graph':
            induct_expr = Batch.to_data_list(induct_expr)

            assert len(induct_expr) == 1
            induct_expr = induct_expr[0]

            labels = ast_def.goal_to_dict(target_goal)['labels']
            induct_expr = induct_expr.to(self.device)
            induct_expr.labels = labels
            tokens = [[t] for t in induct_expr.labels if t[0] == "V"]
            token_inds = [i for i, t in enumerate(induct_expr.labels) if t[0] == "V"]
            if tokens:
                # Encode all nodes in graph
                induct_nodes = self.induct_net(induct_expr)
                # select representations of Variable nodes with ('V' label only)
                token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(self.device))
                target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))

        else:
            tokens = target_goal.split()
            tokens = list(dict.fromkeys(tokens))
            if self.config.data_config.type == 'sequence':
                tokens = [t for t in tokens if t[0] == "V"]
                tokens_ = self.converter(tokens)
                tokens_ = (tokens_[0].to(self.device), tokens_[1].to(self.device))
                if tokens_:
                    token_representations = self.encoder_goal(tokens_).to(self.device)
                    target_representations = einops.repeat(target_representation, '1 d -> n d',
                                                           n=token_representations.shape[0])
                    tokens = [[t] for t in tokens]

            elif self.config.data_config.type == 'fixed':
                tokens = [[t] for t in tokens if t[0] == "V"]
                if tokens:
                    token_representations = self.encoder_goal(tokens).to(self.device)
                    target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))
            else:
                raise NotImplementedError("Induction for non-supported data type")

        if tokens:
            # pass through term_net
            candidates = torch.cat([token_representations, target_representations], dim=1)
            scores = self.term_net(candidates, tac)
            term_probs = F.softmax(scores, dim=0)
            term_m = Categorical(term_probs.squeeze(1))

            if replay_term is None:
                term = term_m.sample()
            else:
                term = torch.tensor([tokens.index(["V" + replay_term])]).to(self.device)

            arg_probs.append(term_m.log_prob(term))
            tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
            tactic = "Induct_on `{}`".format(tm)

        else:
            arg_probs.append(torch.tensor(0))
            tactic = "Induct_on"

        return tactic, arg_probs

    def get_arg_tac(self, target_representation,
                    num_args,
                    encoded_fact_pool,
                    tac,
                    candidate_args,
                    env, replay_arg=None):

        hidden0 = hidden1 = target_representation
        hidden0 = hidden0.to(self.device)
        hidden1 = hidden1.to(self.device)

        hidden = (hidden0, hidden1)

        # concatenate the candidates with hidden states.
        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
        hiddenl = [hc.unsqueeze(0) for _ in range(num_args)]
        hiddenl = torch.cat(hiddenl)

        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
        candidates = candidates.to(self.device)
        input = tac

        # run it once before predicting the first argument
        hidden, _ = self.arg_net(input, candidates, hidden)

        # the indices of chosen args
        arg_step = []
        arg_step_probs = []

        if self.tactic_pool[tac] in self.thm_tactic:
            arg_len = 1
        else:
            arg_len = self.config.arg_len

        for i in range(arg_len):
            hidden, scores = self.arg_net(input, candidates, hidden)
            arg_probs = F.softmax(scores, dim=0)
            arg_m = Categorical(arg_probs.squeeze(1))

            if replay_arg is None:
                arg = arg_m.sample()
            else:
                try:
                    if isinstance(replay_arg, list):
                        name_parser = replay_arg[i].split(".")
                    else:
                        name_parser = replay_arg.split(".")

                    theory_name = name_parser[0][:-6]  # get rid of the "Theory" substring
                    theorem_name = name_parser[1]
                    true_arg_exp = env.reverse_database[(theory_name, theorem_name)]
                except:
                    logging.warning(f"Error in parser {i, replay_arg}")
                    raise Exception

                arg = torch.tensor(candidate_args.index(true_arg_exp)).to(self.device)

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

            # appends both hidden and cell states
            candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
            candidates = candidates.to(self.device)

        tac = tactic_pool[tac]
        arg = [candidate_args[j] for j in arg_step]

        tactic = env.assemble_tactic(tac, arg)
        return tactic, arg_step_probs

    def forward(self, batch, train_mode=True):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        # logging.debug(f"Goal: {goal}")

        encoded_fact_pool = self.encoder_premise(allowed_fact_batch)
        # logging.debug(encoded_fact_pool.shape)

        reward_pool = []
        goal_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0

        max_steps = self.config.max_steps

        for t in range(max_steps):
            target_representation, target_goal, selected_goal, goal_prob = self.get_goal(
                goal_nodes=env.current_goals,
                root=env.graph)

            goal_pool.append(goal_prob)

            tac, tac_prob = self.get_tac(target_representation)

            tac_pool.append(tac_prob)

            if self.tactic_pool[tac] in self.no_arg_tactic:
                tactic = self.tactic_pool[tac]
                arg_probs = [torch.tensor(0)]

            elif self.tactic_pool[tac] == "Induct_on":
                tactic, arg_probs = self.get_term_tac(target_goal=target_goal,
                                                      target_representation=target_representation,
                                                      tac=tac)

            else:
                tactic, arg_probs = self.get_arg_tac(target_representation=target_representation,
                                                     num_args=len(allowed_arguments_ids),
                                                     encoded_fact_pool=encoded_fact_pool,
                                                     tac=tac,
                                                     candidate_args=candidate_args,
                                                     env=env)

            arg_pool.append(arg_probs)
            action = (selected_goal.detach(), tactic)

            try:
                reward, done = env.step(action)
                logging.debug(f"Step taken: {action, reward, done}")
            except Exception:
                logging.warning(f"Step exception: {action, goal}")
                traceback.print_exc()
                return ("Step error", action)

            steps += 1

            if done:
                reward_pool.append(reward)
                if not train_mode:
                    break

                # print (env.history, reward_pool)
                self.proven.append([goal, t + 1])

                if goal in self.replays.keys():
                    if steps < self.replays[goal][0]:
                        self.replays[goal] = deepcopy((steps, env.history))
                else:
                    self.cumulative_proven.append([goal])
                    if env.history is not None:
                        self.replays[goal] = deepcopy((steps, env.history))
                    else:
                        logging.warning(f"History is none. {env.history}")
                break

            elif t == max_steps - 1:
                if not train_mode:
                    break
                reward = -5
                reward_pool.append(reward)

                if goal in self.replays:
                    return self.run_replay(allowed_arguments_ids, candidate_args, env, encoded_fact_pool, goal)
            else:
                reward_pool.append(reward)

        return reward_pool, goal_pool, arg_pool, tac_pool, steps, done

    def get_replay_tac(self, true_tactic_text):
        if true_tactic_text in self.no_arg_tactic:
            true_tac_text = true_tactic_text
            true_args_text = None
        else:
            tac_args = re.findall(r'(.*?)\[(.*?)\]', true_tactic_text)
            tac_term = re.findall(r'(.*?) `(.*?)`', true_tactic_text)
            tac_arg = re.findall(r'(.*?) (.*)', true_tactic_text)
            true_args_text = None
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

    def run_replay(self, allowed_arguments_ids, candidate_args, env, encoded_fact_pool, goal):
        reward_pool = []
        goal_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0

        # goal_history, action_history, reward_history = self.replays[goal][1]
        history = self.replays[goal][1]

        # history_graphs = graph_from_history(goal_history, action_history)

        for t in range(len(history) - 1):

            cur_step = history[t + 1]

            selected_goal, graph = cur_step[0]
            reward = cur_step[1]
            tactic = cur_step[2]

            goal_nodes = nodes_list(graph, result=[])

            goal_idx = [g.goal for g in goal_nodes].index(selected_goal)

            true_fringe = torch.tensor([goal_idx], device=self.device)  # .to(self.device)

            target_representation, target_goal, fringe, goal_prob = self.get_goal(
                goal_nodes=goal_nodes,
                root=graph,
                replay_goal=true_fringe)

            goal_pool.append(goal_prob)
            tac_probs = self.tac_net(target_representation)
            tac_m = Categorical(tac_probs)

            true_tactic_text = tactic

            true_tac_text, true_args_text = self.get_replay_tac(true_tactic_text)

            true_tac = torch.tensor([self.tactic_pool.index(true_tac_text)], device=self.device)  # .to(self.device)
            tac_pool.append(tac_m.log_prob(true_tac))

            assert self.tactic_pool[true_tac.detach()] == true_tac_text

            if self.tactic_pool[true_tac] in self.no_arg_tactic:
                arg_probs = [torch.tensor(0)]

            elif self.tactic_pool[true_tac] == "Induct_on":
                _, arg_probs = self.get_term_tac(target_goal=target_goal,
                                                 target_representation=target_representation,
                                                 tac=true_tac,
                                                 replay_term=true_args_text)

            else:
                _, arg_probs = self.get_arg_tac(target_representation=target_representation,
                                                num_args=len(allowed_arguments_ids),
                                                encoded_fact_pool=encoded_fact_pool,
                                                tac=true_tac,
                                                candidate_args=candidate_args,
                                                env=env,
                                                replay_arg=true_args_text)

            arg_pool.append(arg_probs)

            reward_pool.append(reward)
            steps += 1

        return reward_pool, goal_pool, arg_pool, tac_pool, steps, False

    def save_replays(self):
        torch.save(self.replays, self.replay_dir)

    def get_goal(self, goal_nodes, root, replay_goal=None):
        current_goals = [g.goal for g in goal_nodes]

        reverted = [revert_with_polish(g) for g in current_goals]

        batch = self.converter(reverted)

        if self.config.data_config.type == 'graph':
            batch = batch.to(self.device)

        # sequence case, where batch is (data, attention_mask)
        elif self.config.data_config.type == 'sequence':
            batch = (batch[0].to(self.device), batch[1].to(self.device))

        representations = torch.unsqueeze(self.encoder_goal(batch), 1)

        goal_scores = self.goal_net(representations)

        goal_scores = up_down(goal_scores=goal_scores,
                              goal_nodes=goal_nodes,
                              root=root)

        goal_scores = torch.stack(goal_scores).to(self.device)

        goal_probs = F.softmax(goal_scores, dim=0)

        goal_dist = Categorical(goal_probs)

        if replay_goal is not None:
            goal_idx = replay_goal
        else:
            goal_idx = goal_dist.sample()

        target_goal = goal_nodes[goal_idx].goal

        target_representation = representations[current_goals.index(target_goal)]

        goal_prob = goal_dist.log_prob(goal_idx)

        target_goal = target_goal['polished']['goal']

        return target_representation, target_goal, goal_idx, goal_prob


def up_down(goal_scores, goal_nodes, root):
    # Up Step: set the score of each node as the maximum of itself and all children (i.e. most promising path)

    up_scores = {}

    def up_step(node):
        node_idx = goal_nodes.index(node)

        if node.children == {}:
            up_scores[node_idx] = goal_scores[node_idx]

        else:
            tac_scores = [goal_scores[node_idx]]

            for tac, subgoals in node.children.items():
                sib_scores = []

                for subgoal in subgoals:
                    if goal_nodes.index(subgoal) not in up_scores:
                        up_step(subgoal)

                    assert goal_nodes.index(subgoal) in up_scores

                    sib_scores.append(up_scores[goal_nodes.index(subgoal)])

                tac_score = torch.sum(torch.tensor(sib_scores))

                tac_scores.append(tac_score)

            up_scores[node_idx] = max(tac_scores)

    up_step(root)

    # down step: take product of goal and up_scores from all goals in its context

    up_down_scores = []

    for goal in goal_nodes:
        context_scores = [up_scores[goal_nodes.index(context)] for context in goal.context]
        context_scores.append(goal_scores[goal_nodes.index(goal)])
        context_scores = torch.FloatTensor(context_scores)
        up_down_scores.append(torch.prod(context_scores))

    return up_down_scores