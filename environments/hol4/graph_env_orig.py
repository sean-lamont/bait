import itertools
import os
import signal
from copy import deepcopy
from itertools import count
from sys import exit
from time import sleep

# import plotly.graph_objects as go
# from igraph import Graph
import pexpect
import torch
import torch.nn as nn
from experiments.hol4.rl.exp_config import *

# Updated environment for HOL4 based on graph goal structure (as opposed to fringes in original approach)

HOLPATH = "/home/sean/Documents/hol/HOL/bin/hol --maxheap=256"
# HOLPATH = "/home/sean/Documents/phd/hol/HOL/bin/hol --maxheap=256"
# HOLPATH = "/home/sean/Documents//HOL4/HOL/bin/hol --maxheap=256"












ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def parse_theory(pg):
    theories = re.findall(r'C\$(\w+)\$ ', pg)
    theories = set(theories)
    for th in EXCLUDED_THEORIES:
        theories.discard(th)
    return list(theories)

def list_drop_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def normalize_args(tactic):
    tac_args = re.findall(r'(.*?)\[(.*?)\]', tactic)
    if tac_args:
        tactic_head = tac_args[0][0]
        arglist = tac_args[0][1].split(", ")
        arglist = sorted(arglist, key=len)
        tactic = tactic_head + str(arglist)
        tactic = re.sub("'", "", tactic)
    return tactic


def remove_duplicates(tactic):
    tac_args = re.findall(r'(.*?)\[(.*?)\]', tactic)
    if tac_args:
        tactic_head = tac_args[0][0]
        arglist = tac_args[0][1].split(", ")
        arglist = list_drop_duplicates(arglist)
        tactic = tactic_head + str(arglist)
        tactic = re.sub("'", "", tactic)

    return tactic

def revert_assumptions(context):
    # take a context and return a reverted goal if there are assumptions
    target = context["plain"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in assumptions:
        goal = "(" + i + ")" + " ==> " + "(" + goal + ")"
    return goal


def get_process(pstring):
    pids = []
    fd = os.popen("ps ax | grep " + pstring + " | grep -v grep")
    for line in fd:
        fields = line.split()
        pid = fields[0]
        pids.append(pid)
    fd.close()
    return pids

# class AggChildren(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, children):
#         children.append(x)
#         children = torch.FloatTensor(children)
#         return torch.max(children)
def union_prob(probs):
    n = len(probs)
    result = 0
    for subset_size in range(1, n+1):
        subset = itertools.combinations(probs, subset_size)
        for s in subset:
            # print (s)
            p = 1
            for prob in s:
                # independence assumption
                p *= prob
            if subset_size % 2 == 1:
                result += p
            else:
                result -= p
    return result

class AggChildren(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_score, children_scores):
        children_scores.append(node_score)
        return union_prob(children_scores)


class AggContext(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x, siblings):
        final_score = torch.tensor(1.).to(self.device)
        final_score = x * final_score


        # contexts = torch.FloatTensor([sib.raw_score for sib in siblings])
        # sib_scores = [sib.agg_score for sib in siblings]
        # for score in sib_scores:
        #     final_score *= score

        for sibling in siblings:
            final_score *= sibling.agg_score

        return final_score

class AggSiblings(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, siblings):
        # sib_scores = torch.FloatTensor([sib.raw_score for sib in x])
        score = torch.tensor(1.).to(self.device)

        # scores = [sibling.agg_score for sibling in siblings]
        # for sib_score in scores:
        #     score *= sib_score


        for sibling in siblings:
            score *= sibling.agg_score

        return score

class AggChildrenVec(nn.Module):
    def __init__(self, dim, epsilon = 1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.epsilon = epsilon

    def forward(self, node_vec, child_vecs):

        # node_vec is score vec for parent node

        # x = torch.cat([node_vec, child_vecs], dim = 0)

        x = self.linear(child_vecs)
        x = self.relu(x)
        x = torch.mean(x, dim = 0)

        return node_vec + self.epsilon * x


        # return x


class AggContextVec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, context):#parent, siblings):

        # x should be context vec of parent, siblings should be (num_siblings, dim) for sibling scores

        # x = torch.cat([parent, siblings], dim = 0)
        x = self.linear(context)
        x = self.relu(x)
        x = torch.mean(x, dim = 0)

        return x.unsqueeze(0)

class AggSiblingsVec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, siblings):

        # assume siblings passed as (num_sibs, dim) tensor

        x = self.linear(siblings)
        x = self.relu(x)
        x = torch.mean(x, dim=0)

        return x.unsqueeze(0)


class UpDown():
    def __init__(self, agg_siblings, agg_children, agg_context, device):
        self.device = device
        self.agg_siblings = agg_siblings.to(device)
        self.agg_children = agg_children.to(device)
        self.agg_context = agg_context.to(device)

    # when proved, remove from graph
    def up_step(self, graph):

        if graph.children == {}:
            # if leaf node agg_score will be raw_score
            graph.agg_score = graph.raw_score
            return

        else:
            for tac, val in graph.children.items():
                siblings = val[1]
                for sibling in siblings:
                    self.up_step(sibling)#, score_net)

            # can now assume all children updated

            # tac_scores = []
            for tac, val in graph.children.items():
                # print ("siblings")
                siblings = val[1]
                # set score for this tactic to be sibling aggregation
                graph.children[tac][0] = self.agg_siblings(siblings)

            graph.agg_score = self.agg_children(graph.raw_score, [val[0] for tac, val in graph.children.items()])

            return

    def down_step(self, graph, root=True):
        if root:
            graph.context_score = torch.tensor(1.).to(self.device)

        if graph.children != {}:
            for tac, val in graph.children.items():
                siblings = val[1]
                for sibling in siblings:
                    # print ("context")
                    # print (f"down siblings: {siblings}")
                    # print (f"others {sibling, list(set(siblings) - {sibling})}")
                    # print (f"context before {sibling.context_score}")
                    sibling.context_score = self.agg_context(graph.context_score, list(set(siblings) - {sibling}))
                    # print ("context after ")
                    # print (sibling.context_score)
                    self.down_step(sibling, root=False)
        return

class UpDownVec(nn.Module):
    def __init__(self, device, dim):
        super().__init__()
        self.device = device
        self.agg_siblings = AggSiblingsVec(dim)
        self.agg_children = AggChildrenVec(dim)
        self.agg_context = AggContextVec(dim)
        self.dim = dim

        self.final_fc = torch.nn.Linear(2*dim, dim)
        self.relu = torch.nn.ReLU()
        self.final_head = torch.nn.Linear(dim, 1)

    # assume graph.initial_vec initialised with score_net for all nodes in graph
    def up_step(self, graph):

        if graph.children == {}:
            # if leaf node agg_vec will be agg_children with just initial_vec
            graph.agg_vec = graph.initial_vec
            return

        else:
            for tac, val in graph.children.items():
                siblings = val[1]
                for sibling in siblings:
                    self.up_step(sibling)#, score_net)

            # can now assume all children updated

            # tac_scores = []
            for tac, val in graph.children.items():
                # print ("siblings")
                siblings = val[1]

                # set score for this tactic to be sibling aggregation
                graph.children[tac][0] = self.agg_siblings(torch.cat([s.agg_vec for s in siblings], dim = 0))

            graph.agg_vec = self.agg_children(graph.initial_vec, torch.cat([val[0] for tac, val in graph.children.items()], dim = 0))

            return

    def down_step(self, graph, root=True):
        if root:
            graph.context_vec = torch.ones(self.dim).to(self.device).unsqueeze(0)

        if graph.children != {}:

            for tac, val in graph.children.items():
                siblings = val[1]

                for sibling in siblings:
                    if len(siblings) == 1:
                        sibling.context_vec = self.agg_context(graph.context_vec)
                    else:
                        other_sib_vecs = torch.cat([sib.agg_vec for sib in list(set(siblings) - {sibling})], dim = 0)
                        sibling.context_vec = self.agg_context(torch.cat([graph.context_vec, other_sib_vecs], dim = 0))
                    self.down_step(sibling, root=False)
        return

    # compute the final score following up-down
    def final_agg(self, node):
        x = torch.cat([node.context_vec, node.agg_vec], dim = 1)
        x = self.final_fc(x)
        x = self.relu(x)
        x = self.final_head(x)
        return x


class GoalNode():
    def __init__(self, goal):
        self.goal = goal
        self.from_tac = None
        self.parent = None

        # children in format {tac : [subgoals]}
        self.children = {}

        # self.raw_score = torch.tensor(0.)
        # self.agg_score = torch.tensor(0.)
        # self.context_score = torch.tensor(0.)
        # self.final_score = torch.tensor(0.)

    def prop_proved(self):
        # remove self from parent and
        if self.parent != None:
            self.parent.update_child(self)

    def update_child(self, proven_child):
        # print (proven_child, proven_child.from_tac, self.children)
        # print (self.parent)
        prove_tac = proven_child.from_tac
        # remove proven child from children
        self.children[prove_tac][1].remove(proven_child)

        # if no goals left from same tactic, then this goal is proved
        if self.children[prove_tac][1] == []:
            # if root, then done
            if self.parent == None:
                self.children = {}
                return
            else:
                self.parent.update_child(self)

        return

    def _print(self, depth=1):
        print(depth * "--- " + self.goal["plain"]["goal"])
        if self.from_tac:
            print ("Tac: " + self.from_tac+ "Parent: " + self.parent.goal["plain"]["goal"])
        if len(self.children.keys()) > 0:
            for child in self.children.keys():
                for goal in self.children[child][1]:
                    goal._print(depth + 1)

    def _print_with_scores(self, depth=1):
        print(depth * "--- " + self.goal["plain"]["goal"])
        if self.from_tac:
            print (f"Tac: " + self.from_tac+ " Parent: " + self.parent.goal["plain"]["goal"] + f" scores {self.raw_score, self.agg_score, self.context_score}")
        else:
            print (f"scores: {self.raw_score, self.agg_score, self.context_score}")
        if len(self.children.keys()) > 0:
            for child in self.children.keys():
                for goal in self.children[child][1]:
                    goal._print_with_scores(depth + 1)

# Return set of all unique nodes in a graph
def nodes_list(g, result=[]):
    result.append(g)

    for val in g.children.values():
        siblings = val[1]
        for sibling in siblings:
            nodes_list(sibling, result)

    return list(set(result))


class HolEnv():
    def __init__(self, goal):

        self.handling = None
        self.using = None
        self.frequency = {}
        self.mean_frequency = 0

        self.import_theories = ["probabilityTheory"]
        self.process = pexpect.spawn(HOLPATH, timeout=3)

        # experimental feature
        self.process.delaybeforesend = None
        # import theories
        print("Importing theories...")
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.sendline("val _ = set_trace \"types\" 1;".encode("utf-8"))
        for i in self.import_theories:
            self.process.sendline("load \"{}\";".format(i).encode("utf-8"))
            self.process.sendline("open {};".format(i).encode("utf-8"))
            sleep(3)

        # remove built-in simp lemmas
        # print("Removing simp lemmas...")
        # # self.process.sendline("delsimps [\"HD\", \"EL_restricted\", \"EL_simp_restricted\"];")
        # self.process.sendline("delsimps {};".format(dels))
        # self.process.sendline("delsimps {};".format(dels2))
        # # self.process.sendline("delsimps {};".format(dels3))
        # sleep(1)

        # load utils
        print("Loading modules...")
        self.process.sendline("use \"helper.sml\";")
        # self.process.sendline("val _ = load \"Timeout\";")
        sleep(5)
        print("Configuration done.")
        self.process.expect('\r\n>')
        # self.process.readline()
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))

        # consumes hol4 head
        self.process.expect('\r\n>')

        self.goal = goal

        # try:
        #     self.goal_theory = plain_database[goal][0]
        # except:
        #     self.goal_theory = None

        self.polished_goal = self.get_polish(self.goal)[0]

        self.graph = GoalNode(self.polished_goal)
        self.current_goals = nodes_list(self.graph, result=[])

        # self.current_goals = [self.polished_goal]

        # State stores all current (polished) goals and the tactic labelled edges between them
        #
        # self.fringe = {"content": self.polished_goal,
        #                "parent": None,
        #                "goal": None,
        #                "by_tactic":"",
        #                "reward": None}

        # self.history = [self.fringe]
        # self.history = [self.state]

        # environment history only needs to be list of current goals per step and their parent goals,
        #  as when combined with the action can fully reconstruct proof

        self.history = [[g.goal for g in self.current_goals]]
        # self.history = [[(g.goal, None) for g in self.current_goals]]
        # action should just be goal_node and tactic. Then history can just be goal_node.goal and tactic

        self.action_history = []  # list of tuples (id, id, tactic)

        self.subproofs = {}
        # print("Initialization done. Main goal is:\n{}.".format(self.goal))

    def toggle_simpset(self, mode, theory):
        if mode == "diminish":
            cmd = "val _ = diminish_srw_ss {};".format([theory])
            cmd = re.sub("'", "\"", cmd)
            # print("Removing simp lemmas from {}".format(theory))

        else:
            cmd = "val _ = augment_srw_ss {};".format([theory])
            cmd = re.sub("'", "\"", cmd)
            print("Adding simp lemmas from {}".format(theory))

        reset_cmd = "BasicProvers.recreate_sset_at_parentage (parents \"{}\");".format(theory)
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.sendline(reset_cmd.encode("utf-8"))
        # sleep(0.5)
        # self.process.sendline(cmd.encode("utf-8"))
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.expect('\r\n>')

    def get_names(self, exps):
        # look up the names of exps
        # TODO: rewrite this
        names = []
        for e in exps:
            theorem_name = database[e][1]
            theory_name = database[e][0]
            full_name = theory_name + "Theory." + theorem_name
            names.append(full_name)
        return names

    def assemble_tactic(self, tac, args):
        # args is a list of strings
        if tac in thms_tactic:
            names = self.get_names(args)
            action = tac + re.sub("'", "", str(names))
        elif tac in thm_tactic:
            names = self.get_names(args)
            if names:
                action = tac + " " + names[0]
            else:
                # this will raise an error in HOL4
                action = tac
        else:
            # term tactics will be already assembled
            # no_arg_tactic are handled as is
            action = tac
        return action

    def construct_goal(self, goal):
        s = "g " + "`" + goal + "`;"
        return s

    def construct_tactic(self, tac, limited_time=True):
        if limited_time:
            s = "e " + "(" + tac + ");"
        else:
            s = "unlimited_e " + "(" + tac + ");"
        return s

    def reset(self, new_goal, frequency={}):
        # TODO: record the previous goal

        self.goal = new_goal
        self.polished_goal = self.get_polish(self.goal)[0]

        self.graph = GoalNode(self.polished_goal)
        self.current_goals = nodes_list(self.graph, result=[])


        # self.history = [[(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals]]
        self.history = [[g.goal for g in self.current_goals]]
        # self.history = [[(g.goal, None) for g in self.current_goals]]
        self.action_history = []
        self.subproofs = {}
        self.frequency = frequency

        if self.frequency:
            self.mean_frequency = sum(self.frequency.values()) / len(self.frequency.values())

        print("Initialization done. Main goal is:\n{}.".format(self.goal))

    def close(self):
        pids = get_process("hol")
        pidsh = get_process("buildheap")
        print("Found HOL pids: {}".format(pids))
        for pid in (pids + pidsh):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except:
                pass
            print("Tried closing {}".format(pid))

    def get_polish(self, raw_goal):

        goal = self.construct_goal(raw_goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("top_goals();".encode("utf-8"))
        self.process.expect("val it =")
        self.process.expect([": goal list", ":\r\n +goal list"])
        polished_raw = self.process.before.decode("utf-8")
        polished_subgoals = re.sub("“|”", "\"", polished_raw)
        polished_subgoals = re.sub("\r\n +", " ", polished_subgoals)

        # print("content:{}".format(subgoals))
        # exit()
        pd = eval(polished_subgoals)

        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        data = [{"polished": {"assumptions": e[0][0], "goal": e[0][1]},
                 "plain": {"assumptions": e[1][0], "goal": e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data  # list(zip(pd, [([], raw_goal)]))

    def query(self, raw_goal, tac, limited_time=True):
        # print("content1:{}".format(self.process.before.decode("utf-8")))
        # print("goal is: {}".format(goal))
        # print("tac is: {}".format(tac))
        self.handling = raw_goal
        self.using = tac

        # self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        # self.process.sendline("numSimps.clear_arith_caches();".encode("utf-8"))
        # self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))

        goal = self.construct_goal(raw_goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")

        # bug1 = self.process.before.decode("utf-8")
        # print("bug1: {}".format(bug1))

        tactic = self.construct_tactic(tac, limited_time)
        self.process.sendline(tactic.encode("utf-8"))

        # bug2 = self.process.before.decode("utf-8")
        # print("bug2: {}".format(bug2))

        # Note we may see "metis: proof translation error: trying again with types."]
        try:
            i = self.process.expect(
                ["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof", "Exception",
                 "error"])

        except:
            print("Exception: {} to {} to be debugged".format(tac, raw_goal))
            i = -1

        if i == -1:
            data = "unexpected"
            return data
        # print("i is {}".format(i))

        # bug3 = self.process.before.decode("utf-8")
        # print("bug3: {}".format(bug3))
        # exit()

        # workaround
        while i == 0:
            # skip the proof translation error and read the Exception
            i = self.process.expect(
                ["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof", "Exception",
                 "error"])

            # print("i is {}".format(i))

        if i == 2 or i == 3:
            # bug4 = self.process.before.decode("utf-8")
            # print("bug4: {}".format(bug4))

            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            # bug4 = self.process.before.decode("utf-8")
            # print("bug4: {}".format(bug4))

            try:
                self.process.expect("val it =")
            except:
                print("Exception: {} to {} returned no goals".format(tac, raw_goal))
                exit()

            # this (:\r\n) doesn't seem robust
            self.process.expect([": goal list", ":\r\n +goal list"])
            raw = self.process.before.decode("utf-8")

            # print("sub: {}".format(raw))
            subgoals = re.sub("“|”", "\"", raw)
            subgoals = re.sub("\r\n +", " ", subgoals)
            # print ("q1")
            # get Polished version
            self.process.expect("\r\n>")
            self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            self.process.expect("val it =")
            self.process.expect([": goal list", ":\r\n +goal list"])
            polished_raw = self.process.before.decode("utf-8")
            # print("sub: {}".format(raw))
            polished_subgoals = re.sub("“|”", "\"", polished_raw)
            polished_subgoals = re.sub("\r\n +", " ", polished_subgoals)

            # print ("q2")
            # print("content:{}".format(subgoals))
            # exit()
            # escape colored characters
            polished_subgoals = ansi_escape.sub('', polished_subgoals)
            subgoals = ansi_escape.sub('', subgoals)

            # print ("q3")
            pd = eval(polished_subgoals)
            d = eval(subgoals)
            # data = list(zip(pd, d))
            data = zip(pd, d)
            data = [{"polished": {"assumptions": e[0][0], "goal": e[0][1]},
                     "plain": {"assumptions": e[1][0], "goal": e[1][1]}}
                    for e in data]
            # data = (pd, d)
            # data = eval(subgoals)
        elif i == 1:
            data = []
        elif i == 4:
            j = self.process.expect(["Time", pexpect.TIMEOUT], timeout=0.01)
            if j == 0:
                data = "timeout"
            else:
                # print("pexpect timeout")
                data = "exception"
        else:
            # if PRINT_EXCEPTION:
            #     print("Exception: {} to {}.".format(tac, raw_goal))
            data = "exception"

        # clear stack and consume the remaining stdout
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        return data

    # Returns reward, done
    def step(self, action):

        # print (f"stepping with {action}")
        # if action in self.action_history:
        #     reward = -1
        #     print ("same action")
            # return reward, False  # TODO: make this reward zero?

        # action is now 2 tuple instead of 3, only needing to specify goal_id and tactic

        goal_node, tactic = action


        # print (goal_node.goal, tactic)
        # print (self.action_history)

        if (goal_node.goal, tactic) in self.action_history:
            self.action_history.append((action[0].goal, action[1]))
            self.history.append([g.goal for g in self.current_goals])
            reward = -1

            # print ("Duplicate action!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return reward, False

        target = goal_node.goal["plain"]

        # print (f"Stepping with {goal_node.goal['plain']}, {tactic}")


        if target["assumptions"]:
            # there are assumptions
            goal = revert_assumptions(goal_node.goal)
            d = self.query(goal, "rpt strip_tac >> " + tactic)

        else:
            goal = target["goal"]
            d = self.query(goal, tactic)

        if d == "unexpected":
            self.action_history.append((action[0].goal, action[1]))
            self.history.append([g.goal for g in self.current_goals])
            # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
            reward = UNEXPECTED_REWARD

        elif d != "exception" and d != "timeout":

            # progress has been made
            if [goal_node.goal] != d:

                self.action_history.append((action[0].goal, action[1]))
                # self.action_history.append(action)

                if d == []:

                    if goal_node == self.graph:
                        # print ("Goal proved")
                        self.history.append([[]]) #goal_node.parent.goal if goal_node.parent is not None else None))
                        return 5, True

                    reward = 0.2



                    # print (f"subgoal proved {goal_node.goal['plain']}")

                    # print ("before prop")
                    # self.graph._print()
                    goal_node.prop_proved()
                    # print ("after prop")
                    # self.graph._print()

                    self.current_goals = nodes_list(self.graph, result=[])


                    # if original goal proved in prop_proved, then all children set to {}, so check for this then return if parent proved
                    # print (self.current_goals, len(self.current_goals))
                    if len(self.current_goals) == 1:
                        # print ("subgoal proved, proving original")
                        self.history.append([[]]) #goal_node.parent.goal if goal_node.parent is not None else None))
                        # self.history.append([([], goal_node.parent.goal if goal_node.parent is not None else None)])

                        return 5, True

                    hist = [g.goal for g in self.current_goals]
                    hist.append([])
                    self.history.append(hist)
                    # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])

                else:

                    # same tactic
                    if tactic in goal_node.children.keys():
                        # print ("tac duplicate")
                        # self.action_history.append(action self.action_history.append((action[0].goal, action[1]))
                        self.history.append([g.goal for g in self.current_goals])
                        reward = -0.1
                        return reward, False

                     # same subgoal(s) as previous tactic for same node
                    subgoal_list = []
                    for val in goal_node.children.values():
                        goals = set(((g.goal['plain']['goal'], tuple(g.goal['plain']['assumptions'])) for g in val[1]))
                        subgoal_list.append(goals)


                    if set(((d_['plain']['goal'], tuple(d_['plain']['assumptions'])) for d_ in d)) in subgoal_list:
                        # print ("subgoal duplicate")
                        # print (f"{[d_['plain'] for d_ in d]}")
                        # self.action_history.append(action)
                        # self.action_history.append((action[0].goal, action[1]))
                        self.history.append([g.goal for g in self.current_goals])
                        # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
                        reward = -0.1
                        return reward, False


                    # bug? where multiple copies of identical subgoal are added.

                    strd = [str(d_) for d_ in d]

                    if len(list(set(strd))) < len(strd):
                        # print ("HOL duplicate subgoal?")
                        # print (f"{[d_['plain'] for d_ in d]}")
                        self.history.append([g.goal for g in self.current_goals])
                        # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
                        reward = -0.1
                        return reward, False

                    reward = 0.1

                    # print (f"Adding subgoals {[d_['plain'] for d_ in d]}")


                    for i, goal in enumerate(d):
                        new_node = GoalNode(goal)
                        new_node.parent = goal_node
                        new_node.from_tac = tactic

                        if tactic in goal_node.children.keys():
                            goal_node.children[tactic][1].append(new_node)
                        else:
                            # child dict has {tac : (score, [children])}
                            goal_node.children[tactic] = [None,  [new_node]]


                    self.current_goals = nodes_list(self.graph, result=[])
                    self.history.append([g.goal for g in self.current_goals])
                    # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])

                    # print (f"Subgoals {[d_['plain'] for d_ in d]}")
                    # print (f"History {self.history}")
                    # self.history = [(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals]

            else:
                # nothing changed
                reward = -0.1
                # self.action_history.append(action)
                self.action_history.append((action[0].goal, action[1]))
                self.history.append([g.goal for g in self.current_goals])
                # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
                # self.history = [(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals]
        else:
            if d == "timeout":
                reward = -0.1
                # self.action_history.append(action)
                self.action_history.append((action[0].goal, action[1]))
                self.history.append([g.goal for g in self.current_goals])
                # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
                # self.history = [(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals]
            else:
                reward = -0.1
                self.action_history.append((action[0].goal, action[1]))
                self.history.append([g.goal for g in self.current_goals])
                # self.history.append([(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals])
                # self.history = [(g.goal, g.parent.goal) if g.parent is not None else (g.goal, None) for g in self.current_goals]

        return reward, False


def extract_proof(history):
    qed = history[-1]
    path = []
    proof = []
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    for _ in count():
        parent_fringe = history[parent_fringe_id]
        path.append(history[parent_fringe_id])
        parent_goal = parent_fringe["content"][parent_goal_id]
        plain_text_goal = parent_goal["plain"]["goal"]
        proof.append((plain_text_goal, tactic))

        if parent_fringe_id == 0:
            proof.reverse()
            return proof

        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]


def annotate_origin(history):
    for fid, f in enumerate(history):
        parent_fringe_id = f["parent"]

        parent_goal_id = f["goal"]
        if not fid:
            f["content"][0]["origin_fringe"] = 0
        else:
            parent_fringe = history[parent_fringe_id]
            # parent_goal = parent_fringe["content"]["parent_goal_id"]
            # new_content = f["content"][len(parent_fringe["content"])-1:]
            for i, e in enumerate(f["content"]):
                if i < len(parent_fringe["content"]) - 1:
                    # parent_goal_id is the id of the removed parent goal
                    if i < parent_goal_id:
                        e["origin_fringe"] = parent_fringe["content"][i]["origin_fringe"]
                    else:
                        e["origin_fringe"] = parent_fringe["content"][i + 1]["origin_fringe"]
                else:
                    e["origin_fringe"] = fid

    return history


def construct_map(history):
    history = annotate_origin(history)

    dependency_table = {}
    qed = history[-1]
    path = []
    proof = []
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    content = qed["content"]

    for _ in count():
        parent_fringe = history[parent_fringe_id]
        parent_goal = parent_fringe["content"][parent_goal_id]
        plain_text_goal = parent_goal["plain"]["goal"]
        plain_text_assumptions = parent_goal["plain"]["assumptions"]
        # two id's to distinguish different goals that have same expression
        # dependency_table[(revert_assumptions(parent_goal), parent_fringe_id, parent_goal_id)] = (tactic, [(p, current_fringe_id, content.index(p)) for p in content if p not in parent_fringe["content"]])
        new_content = content[len(parent_fringe["content"]) - 1:]
        dependency_table[(revert_assumptions(parent_goal), parent_goal["origin_fringe"])] = (
        remove_duplicates(tactic), new_content)

        if parent_fringe_id == 0:
            break

        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]

    # script = generate_script(content[parent_fringe_id]["plain"]["goal"], dependency_table)
    return dependency_table


def generate_script(key, m):
    script, subgoals = m[key]
    if len(subgoals) >= 2:
        for i in subgoals:
            new_key = (revert_assumptions(i), i["origin_fringe"])
            if i["plain"]["assumptions"]:
                script = script + " >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> {})".format(
                    generate_script(new_key, m))
            else:
                script = script + " >- ({})".format(generate_script(new_key, m))
    elif len(subgoals) == 1:
        new_key = (revert_assumptions(subgoals[0]), subgoals[0]["origin_fringe"])
        if subgoals[0]["plain"]["assumptions"]:
            script = script + " >> (rpt (pop_assum mp_tac) >> rpt strip_tac >> {})".format(generate_script(new_key, m))
        else:
            script = script + " >> {}".format(generate_script(new_key, m))
    else:
        pass
    return script


def reconstruct_proof(history):
    history = annotate_origin(history)
    m = construct_map(history)
    key = (history[0]["content"][0]["plain"]["goal"], 0)
    script = generate_script(key, m)
    return script


def check_proof(env, history):
    script = reconstruct_proof(history)
    goal = history[0]["content"][0]["plain"]["goal"]
    data = env.query(goal, script, False)
    return (data == [])


def extract_path_id(history):
    qed = history[-1]
    # print(qed)
    path = [-1]
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    for _ in count():
        parent_fringe = history[parent_fringe_id]
        path.append(parent_fringe_id)

        if parent_fringe_id == 0:
            return path

        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]


def extract_proof_replay(history):
    # this is used to replay a successful proof without querying HOL
    qed = history[-1]
    path = []
    replay = []
    parent_fringe_id = qed["parent"]
    parent_goal_id = qed["goal"]
    tactic = qed["by_tactic"]
    for _ in count():
        parent_fringe = history[parent_fringe_id]
        replay.append((parent_fringe_id, parent_goal_id, tactic))
        if parent_fringe_id == 0:
            replay.reverse()
            return replay

        content = parent_fringe["content"]
        parent_fringe_id = parent_fringe["parent"]
        parent_goal_id = parent_fringe["goal"]
        tactic = parent_fringe["by_tactic"]


def get_text(fringe):
    # [((polished assumptions, polished goal),(assumptions, goal)),...]
    # t = [p[1] for p in fringe]
    # texts = []
    if not fringe:
        return "QED"

    text = ""
    for i, p in enumerate(fringe):
        text += "{}: {}<br>".format(i, p["plain"])

    return text[:-4]


def make_tree(history):
    es = []
    for i in history:
        p = i["parent"]
        if p != None:  # p can be 0
            es.append(((p, history.index(i)),  # edge
                       (i["goal"], i["by_tactic"])))  # label

    return es


def draw_tree(history):  # , output_graph=False):
    nv = len(history)
    eslb = make_tree(history)
    es = [i[0] for i in eslb]
    g = Graph(nv, es, True)

    g.vs["goals"] = [get_text(i["content"]) for i in history]
    g.es["by applying tactic on"] = [i[1] for i in eslb]
    g.es["by applying tactic on"] = ["Step: {}<br>Target: {}<br>Tactic: {}".format(n, i[1][0], i[1][1]) for (n, i) in
                                     enumerate(eslb)]
    g.vs["label"] = g.vs["goals"]
    g.es["label"] = g.es["by applying tactic on"]
    # g.add_vertices(nv)
    # g.add_edges(es)

    # if output_graph:
    #     layout = g.layout("rt")
    #     plot(g, layout = layout, bbox = (1024, 1024))
    # print(g)
    # return summary(g)

    lay = g.layout('rt')

    position = {k: lay[k] for k in range(nv)}
    Y = [lay[k][1] for k in range(nv)]
    M = max(Y)

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in es:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    # fs = [get_text(i["content"]) for i in history]

    vlabels = [get_text(i["content"]) for i in history]
    elabels = g.es["by applying tactic on"]

    # calculate the middle points for edge labels
    Xel = []
    Yel = []
    for edge in es:
        Xel += [0.5 * (position[edge[0]][0] + position[edge[1]][0])]
        Yel += [0.5 * (2 * M - position[edge[0]][1] + 2 * M - position[edge[1]][1])]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines+text',
                             name='Actions',
                             line=dict(color='rgb(0,0,0)', width=1),
                             # text=elabels,
                             # hoverinfo='text',
                             opacity=0.8
                             # hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='Dead',
                             marker=dict(symbol='circle-dot',
                                         size=18,
                                         color='#FF0000',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=vlabels,
                             hoverinfo='text',
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=Xel,
                             y=Yel,
                             mode='markers',
                             name='Action labels',
                             marker=dict(color='rgb(210,210,210)',
                                         size=1),
                             text=elabels,
                             hoverinfo='text'
                             ))
    if "QED" in g.vs["goals"]:
        path = extract_path_id(history)
        pathXn = [Xn[i] for i in path]
        pathYn = [Yn[i] for i in path]
        plabels = [vlabels[i] for i in path]

        fig.add_trace(go.Scatter(x=pathXn,
                                 y=pathYn,
                                 mode='markers',
                                 name='Path',
                                 marker=dict(symbol='circle-dot',
                                             size=18,
                                             color='#6175c1',  # '#DB4551',
                                             line=dict(color='rgb(50,50,50)', width=1)
                                             ),
                                 text=plabels,
                                 hoverinfo='text',
                                 opacity=0.8
                                 ))

    # fig.show()
    fig.write_html('first_figure.html', auto_open=True)


def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs


# def encode(s):
#     # s is a string
#     # print("Encoding: {}".format(s))
#     s = s.split()
#     r = []
#     for c in s:
#         if c not in dictionary:
#             dictionary[c] = len(dictionary) + 1
#         r.append(dictionary[c])
#     # pad r with 0's
#     r = (r + MAX_LEN * [0])[:MAX_LEN]

#     # a list whose length is max_len
#     return r

# def encode_goal(goal): # goal is an (assumption, goal) pair
#     # create an empty entry
#     unit = MAX_LEN * [0]
#     target = goal["polished"] # the polished form
#     polished_goal = target["goal"]
#     context = [encode(polished_goal)]
#     for i in range(MAX_ASSUMPTIONS):
#         if i < len(target["assumptions"]):
#             context.append(encode(target["assumptions"][i]))
#         else:
#             context.append(unit)

#     # returns the first polished goal and the encoding of the goal-assumptions pair
#     return (polished_goal, torch.FloatTensor(context))


# def context_encoder(context): # a context is an (assumption, goal) pair
#     # create an empty entry
#     unit = MAX_LEN * [0]
#     target = context["polished"] # the polished form
#     polished_goal = target["goal"]
#     context = [encode(polished_goal)]
#     for i in range(MAX_ASSUMPTIONS):
#         if i < len(target["assumptions"]):
#             context.append(encode(target["assumptions"][i]))
#         else:
#             context.append(unit)

#     # returns the first polished goal and the encoding of the goal-assumptions pair
#     return torch.FloatTensor(context)


# def gather_goals(history):
#     fringe_sizes = []
#     goals = []
#     for i in history:
#         c = i["content"]
#         goals.extend(c)
#         fringe_sizes.append(len(c))
#     return goals, fringe_sizes


# def gather_encoded_content(history):
#     fringe_sizes = []
#     goals = []
#     for i in history:
#         c = i["content"]
#         goals.extend(c)
#         fringe_sizes.append(len(c))
#     representations = []
#     for g in goals:
#         _, encoded = encode_goal(g)
#         representations.append(encoded.unsqueeze(0))
#     return torch.stack(representations), goals,fringe_sizes

# def revert_with_polish(context):
#    target = context["polished"]
#    assumptions = target["assumptions"]
#    goal = target["goal"]
#    for i in reversed(assumptions): 
#        goal = "@ @ Dmin$==> {} {}".format(i, goal)
#    return goal    

def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions):
        # goal = "@ @ D$min$==> {} {}".format(i, goal)
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)

    return goal


# def gather_encoded_content(history, encoder):
#    # figure out why this is slower than tests
#    # figured out: remember to do strip().split()
#    fringe_sizes = []
#    contexts = []
#    reverted = []
#    for i in history:
#        c = i["content"]
#        contexts.extend(c)
#        fringe_sizes.append(len(c))
#    for e in contexts:
#        g = revert_with_polish(e)
#        reverted.append(g.strip().split())
#    # print(reverted)
#    # s1 = timeit.default_timer()
#    out, sizes = batch_encoder.encode(reverted)
#    # merge two hidden variables
#    representations = torch.cat(out.split(1), dim=2).squeeze(0)
#    # print(representations.shape)
#    # s2 = timeit.default_timer()    
#    # print(s2-s1)
#
#    return representations, contexts, fringe_sizes

def gather_encoded_content(history, encoder):
    fringe_sizes = []
    contexts = []
    reverted = []
    for i in history:
        c = i["content"]
        contexts.extend(c)
        fringe_sizes.append(len(c))
    for e in contexts:
        g = revert_with_polish(e)
        reverted.append(g.strip().split())
    out = []
    sizes = []
    for goal in reverted:
        out_, sizes_ = encoder.encode([goal])
        out.append(torch.cat(out_.split(1), dim=2).squeeze(0))
        sizes.append(sizes_)

    representations = out

    return representations, contexts, fringe_sizes


def split_by_fringe(goal_set, goal_scores, fringe_sizes):
    # group the scores by fringe
    fs = []
    gs = []
    counter = 0
    for i in fringe_sizes:
        end = counter + i
        fs.append(goal_scores[counter:end])
        gs.append(goal_set[counter:end])
        counter = end
    return gs, fs


def construct_new_map(history, action_history):
    map = {}

    assert (len(history) == len(action_history) + 1)
    # print ([[h[0]['plain']['goal'] for h in history] for hist in history])

    for id, goals in enumerate(history[:-1]):

        new_goals = [g for g in history[id + 1] if g not in history[id]]

        goal, tactic = action_history[id]

        # print (goal['plain']['goal'], tactic)
        # print ([g[0]['plain']['goal'] if g[0] != [] else [] for g in new_goals])


        # edge case when two goals proven in immediate succession

        if [] in history[id + 1] and [] in history[id]:
            new_goals = [[]]

        if new_goals:

            if revert_assumptions(goal) in map.keys():

                if map[revert_assumptions(goal)] is None:

                    map[revert_assumptions(goal)] = [(remove_duplicates(tactic), [g for g in new_goals])]#[g[0] if g[0] != [] else [] for g in new_goals])]

                else:

                    map[revert_assumptions(goal)].append((remove_duplicates(tactic), [g for g in new_goals]))#[g[0] if g[0] != [] else [] for g in new_goals]))
            else:
                map[revert_assumptions(goal)] = [(remove_duplicates(tactic), [g for g in new_goals])]#[g[0] if g[0] != [] else [] for g in new_goals])]

            # add empty entry for subgoals as well
            for new_goal in new_goals:
                if new_goal != []:
                    if revert_assumptions(new_goal) not in map.keys():
                        map[revert_assumptions(new_goal)] = None

        # no new goals generated for goal
        else:
            if revert_assumptions(goal) not in map.keys():
                # map[revert_assumptions(goal)].append((remove_duplicates(tactic), None))
            # else:
                map[revert_assumptions(goal)] = None # [(remove_duplicates(tactic), None)]

    return map


# def generate_script_new(key, map):
#     # print (map.keys())
#     script = ""
#
#     if map[key] is None:
#         return script
#
#     for step in map[key]:
#
#         script, subgoals = step#map[key]
#
#         if len(subgoals) >= 2:
#             for i in subgoals:
#                 new_key = revert_assumptions(i)
#                 if i["plain"]["assumptions"]:
#                     script = script + " >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> {})".format(
#                         generate_script_new(new_key, map))
#                 else:
#                     script = script + " >- ({})".format(generate_script_new(new_key, map))
#
#         elif len(subgoals) == 1:
#
#             # if done
#             if subgoals[0] == []:
#                 # print (f"SDFSDFSDFSDFSDF {script}")
#                 continue
#
#             new_key = revert_assumptions(subgoals[0])
#             if subgoals[0]["plain"]["assumptions"]:
#                 script = script + " >> (rpt (pop_assum mp_tac) >> rpt strip_tac >> {})".format(generate_script_new(new_key, map))
#             else:
#                 script = script + " >> {}".format(generate_script_new(new_key, map))
#         else:
#             pass
#     return script


def graph_from_history(history, action_history):
    graphs = []

    main_goal = history[0][0]

    initial_graph = GoalNode(main_goal)

    graphs.append(initial_graph)

    for i in range(1, len(history) - 1):

        next_graph = deepcopy(graphs[i - 1])

        chosen_goal, tactic = action_history[i - 1]

        # print (f"Chosen: {chosen_goal}")
        current_goals = nodes_list(next_graph, result=[])

        # print (f"Current goals {[g.goal for g in current_goals]}")


        # goal_idx = current_goals.index(chosen_goal)

        goal_idx = [g.goal for g in current_goals].index(chosen_goal)

        chosen_node = current_goals[goal_idx]

        # edge case when two goals proven in immediate succession

        if [] in history[i] and [] in history[i - 1]:
            # proven goal/subgoal
            chosen_node.prop_proved()
            graphs.append(next_graph)
            continue

        new_goals = [g for g in history[i] if g not in history[i - 1]]
        # print (f"New goals: {new_goals}")



        # no progress made
        if not new_goals:
            graphs.append(next_graph)
            continue

        if new_goals[0] == [] and len(new_goals) == 1:
            # proven goal/subgoal
            chosen_node.prop_proved()
            graphs.append(next_graph)
            continue

        first_child = GoalNode(new_goals[0])
        first_child.parent = chosen_node
        first_child.from_tac = tactic

        chosen_node.children[tactic] = [None, [first_child]]

        for subgoal in new_goals[1:]:
            next_node = GoalNode(subgoal)
            next_node.parent = chosen_node
            next_node.from_tac = tactic
            chosen_node.children[tactic][1].append(next_node)

        graphs.append(next_graph)

    return graphs





# todo add and return best history with this corresponding to the found proof
#
def find_best_proof(goal, map):
    found_proofs = []

    if map[goal] is None:
        return None

    # print ("a")
    for (tac, subgoals) in map[goal]:
        subgoal_proofs = []

        if len(subgoals) >= 2:

            # print ("b")
            script = tac
            subgoal_steps = 0
            for subgoal in subgoals:

                # print ("c")
                new_key = revert_assumptions(subgoal)
                # print ("d")
                proof = find_best_proof(new_key, map)

                # print ("dd")
                # need every proof from subgoal, so if none exist then break
                if proof is None:
                    break

                # print ("e")
                if subgoal['plain']['assumptions']:
                    script = script + f" >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> {proof[1]})"
                else:
                    script = script + f" >- ({proof[1]})"

                # print ("f")
                subgoal_steps += proof[0]
                subgoal_proofs.append((proof[0], script))

                # print ("g")
            else:

                # all subgoals done
                found_proofs.append((1 + subgoal_steps, script))
                # found_proof = []
                # print ("h")
                # for proof in subgoal_proofs:
                #     found_proof.append((1 + proof[0], tac + proof[1]))

                # print ("i")
        elif len(subgoals) == 1:

            # if subgoals is None:
            #     continue
            # if proof done
            # print ("j")
            if subgoals[0] == []:
                proof = (1, tac)
                found_proofs.append(proof)
                continue

            # print ("k")
            new_key = revert_assumptions(subgoals[0])
            proof = find_best_proof(new_key, map)

            # print ("l")
            if proof is None:
                continue

            # print ("m")
            if subgoals[0]['plain']['assumptions']:
                script = f" >> (rpt (pop_assum mp_tac) >> rpt strip_tac >> {proof[1]})"
            else:
                script = f" >> {proof[1]}"

            # print ("n")
            found_proofs.append((1 + proof[0], tac + script))

            # print ("o")

    if not found_proofs:
        # print ("none found")
        return None

    # print (found_proofs)

    best_val = min([p[0] for p in found_proofs])

    best_idx = [p[0] for p in found_proofs].index(best_val)

    return (best_val, found_proofs[best_idx][1])






# import pickle
# with open("paper_goals.pk", "rb") as f:
#     goals = pickle.load(f)
#
# env = HolEnv(goals[0][1])
# print (env.query(env.graph.goal[0]['plain']['goal'], 'Induct_on `s1`'))
# print (len(env.current_goals))
# print (env.current_goals)
# env.step((0, 'Induct_on `s1`'))
# print (len(env.current_goals))
# print (env.current_goals)
# print (env.current_goals[1].parent)
# print (env.graph.children)
