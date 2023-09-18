import json
import logging

import plotly.graph_objects as go
from igraph import Graph
import pexpect
import torch
from itertools import count
from time import sleep
import signal
import os
from copy import deepcopy
import re


#todo move to config
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

UNEXPECTED_REWARD = -10
HOLPATH = "environments/hol4/HOL/bin/hol --maxheap=256"
EXCLUDED_THEORIES = ["min"]

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


class HolEnv:
    def __init__(self, goal):
        with open("data/hol4/data/adjusted_db.json") as f:
            self.database = json.load(f)

        self.reverse_database = {(value[0], value[1]) : key for key, value in self.database.items()}

        self.handling = None
        self.using = None
        self.frequency = {}
        self.mean_frequency = 0
        
        self.import_theories = ["probabilityTheory"]
        self.process = pexpect.spawn(HOLPATH, timeout=3)
        
        # experimental feature
        self.process.delaybeforesend = None

        # import theories
        # logging.debug("Importing theories...")

        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.sendline("val _ = set_trace \"types\" 1;".encode("utf-8"))
        for i in self.import_theories:
            self.process.sendline("load \"{}\";".format(i).encode("utf-8"))
            self.process.sendline("open {};".format(i).encode("utf-8"))
            sleep(5)

        # remove built-in simp lemmas
        # logging.debug("Removing simp lemmas...")
        # # self.process.sendline("delsimps [\"HD\", \"EL_restricted\", \"EL_simp_restricted\"];")
        # self.process.sendline("delsimps {};".format(dels))
        # self.process.sendline("delsimps {};".format(dels2))
        # # self.process.sendline("delsimps {};".format(dels3))
        # sleep(1)
        
        # load utils
        # logging.debug("Loading modules...")
        self.process.sendline("use \"helper.sml\";")
        # self.process.sendline("val _ = load \"Timeout\";")
        sleep(5)
        # logging.debug("Configuration done.")
        self.process.expect('\r\n>')
        # self.process.readline()
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))


        # consumes hol4 head
        self.process.expect('\r\n>')

        self.goal = goal

        self.polished_goal = self.get_polish(self.goal)

        self.fringe = {"content": self.polished_goal,
                       "parent": None,
                       "goal": None,
                       "by_tactic":"",
                       "reward": None}

        # a fringe is a list of the form
        # [((polished assumptions, polished goal),
        #   (assumptions, goal)),
        #   ...]

        self.history = [self.fringe]
        self.action_history = [] # list of tuples (id, id, tactic)
        self.subproofs = {}
        logging.debug("Initialization done. Main goal is:\n{}.".format(self.goal))

    def toggle_simpset(self, mode, theory):
        if mode == "diminish":
            cmd = "val _ = diminish_srw_ss {};".format([theory])
            cmd = re.sub("'", "\"", cmd)
            logging.debug("Removing simp lemmas from {}".format(theory))
            
        else:
            cmd = "val _ = augment_srw_ss {};".format([theory])
            cmd = re.sub("'", "\"", cmd)
            logging.debug("Adding simp lemmas from {}".format(theory))
            
        # self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        # # sleep(0.5)
        # self.process.sendline(cmd.encode("utf-8"))
        # self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        # self.process.expect('\r\n>')

        reset_cmd = "BasicProvers.recreate_sset_at_parentage (parents \"{}\");".format(theory)
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.sendline(reset_cmd.encode("utf-8"))
        # sleep(0.5)
        # self.process.sendline(cmd.encode("utf-8"))
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.expect('\r\n>')

    def get_names(self, exps):
        # look up the names of exps
        names = []
        for e in exps:
            theorem_name = self.database[e][1]
            theory_name = self.database[e][0]
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

        logging.debug("Resetting goal to be {}".format(self.goal))

        self.polished_goal = self.get_polish(self.goal)

        self.fringe = {"content": self.polished_goal,
                       "parent": None,
                       "goal": None,
                       "by_tactic":"",
                       "reward": None}

        self.history = [self.fringe]

        self.action_history = []
        self.subproofs = {}
        self.frequency = frequency

        if self.frequency:
            self.mean_frequency = sum(self.frequency.values())/len(self.frequency.values())

        logging.debug("Initialization done. Main goal is:\n{}.".format(self.goal))

    def close(self):    
        pids = get_process("hol")
        pidsh = get_process("buildheap")
        print("Found HOL pids: {}".format(pids))
        for pid in (pids+pidsh):
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
        polished_subgoals = re.sub("“|”","\"", polished_raw)
        polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

        # print("content:{}".format(subgoals))
        # exit()
        pd = eval(polished_subgoals)
        
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                 "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]
        return data
    
    def query(self, raw_goal, tac, limited_time=True):
        # print("content1:{}".format(self.process.before.decode("utf-8")))
        # print("goal is: {}".format(raw_goal))
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
            # print("Exception: {} to {} to be debugged".format(tac, raw_goal))
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
            i = self.process.expect(["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof" , "Exception", "error"])

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
                logging.debug("Exception: {} to {} returned no goals".format(tac, raw_goal))
                return "exception"
                # exit()
            
            # this (:\r\n) doesn't seem robust
            self.process.expect([": goal list", ":\r\n +goal list"])
            raw = self.process.before.decode("utf-8")
            
            # print("sub: {}".format(raw))
            subgoals = re.sub("“|”","\"", raw)
            subgoals = re.sub("\r\n +"," ", subgoals)

            # get Polished version
            self.process.expect("\r\n>")
            self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            self.process.expect("val it =")
            self.process.expect([": goal list", ":\r\n +goal list"])
            polished_raw = self.process.before.decode("utf-8")         
            # print("sub: {}".format(raw))
            polished_subgoals = re.sub("“|”","\"", polished_raw)
            polished_subgoals = re.sub("\r\n +"," ", polished_subgoals)

            # print("content:{}".format(subgoals))
            # exit()
            # escape colored characters
            polished_subgoals = ansi_escape.sub('', polished_subgoals)
            subgoals = ansi_escape.sub('', subgoals)

            pd = eval(polished_subgoals)
            d = eval(subgoals)
            # data = list(zip(pd, d))
            data = zip(pd, d)
            data = [{"polished":{"assumptions": e[0][0], "goal":e[0][1]},
                     "plain":{"assumptions": e[1][0], "goal":e[1][1]}}
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

    def step(self, action):            
        if action in self.action_history:
            reward = -1
            return reward, False # TODO: make this reward zero?

        fringe_id, goal_id, tactic = action
        target_fringe = self.history[fringe_id]
        pre_target = target_fringe["content"][goal_id]
        target = pre_target["plain"]
        # tactic = normalize_args(tactic)

        if target["assumptions"]:
            # there are assumptions
            goal = revert_assumptions(pre_target)
            d = self.query(goal, "rpt strip_tac >> " + tactic)
        else:
            # no assumptions
            goal = target["goal"]
            d = self.query(goal, tactic)

        if d == "unexpected":
            reward = UNEXPECTED_REWARD

        elif d != "exception" and d != "timeout":
            # progress has been made
            if [pre_target] != d:
                new_content = deepcopy(target_fringe["content"])
                new_content.remove(pre_target)
                new_content.extend(d)

                # never any repeats without a dupe (all new subgoals exist in an old fringe, if the current fringe equals an old fringe)
                # but there are sometimes dupes without repeat. I.e. all new subgoals exist in an old fringe, but the current fringe does not equal that fringe
                for f in self.history:
                    # if results in something occurred before
                    if new_content == f["content"]:
                        # print ('repeat\n')
                        return -0.1, False # do not penalize the agent here
                        # return 0, False # do not penalize the agent here

                coordinate = (fringe_id, goal_id)
                if coordinate in self.subproofs:
                    one_step = {"subgoals": d,"via": tactic}
                    current_branches = self.subproofs[coordinate]
                    current_branches.append(one_step)
                else:
                    self.subproofs[coordinate] = [{"subgoals": d,"via": tactic}]
                    
                new_fringe = {"content": new_content,
                              "parent": fringe_id,
                              "goal": goal_id,
                              "by_tactic": tactic,
                              "reward": None}
                reward = 0.1 # *(2 - tac_len * 0.2)
                self.action_history.append(action)
                
                # reward solving a subgoal
                if d == []:
                    reward = 0.2

                if new_content == []:
                    new = self.goal
                    
                    # shape reward
                    if self.frequency:
                        self.frequency[self.goal] += 1
                        # mean_frequency = sum(self.frequency.values())/len(self.frequency.values())
                        if self.frequency[self.goal] >= self.mean_frequency:
                            reward = 5 # make this 200?
                        else:
                            reward = 15 # * (1 + mean_frequency - self.frequency[self.goal])
                    else:
                        reward = 5

                    # new_fringe.update({"reward": reward})
                    new_fringe["reward"] = reward
                    self.history.append(new_fringe)
                    return reward, True
                
                new_fringe["reward"] = reward
                self.history.append(new_fringe)
                
            else:
                # nothing changed
                reward = -0.1
                self.action_history.append(action)
        else:
            if d == "timeout":
                reward = -0.1
                self.action_history.append(action)
            else:
                # not applicable
                reward = -0.1
                self.action_history.append(action)
                
        return reward, False


    def gen_fact_pool(self, goal):
        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', goal[0])))
        goal_theory = self.database[goal[0]][0]
        polished_goal = self.fringe["content"][0]["polished"]["goal"]
        try:
            allowed_arguments_ids = []
            candidate_args = []
            for i, t in enumerate(self.database):
                theory_allowed = self.database[t][0] in allowed_theories
                diff_theory = self.database[t][0] != goal_theory
                prev_theory = int(self.database[t][3]) < int(self.database[polished_goal][3])
                if theory_allowed and (diff_theory or prev_theory):
                    allowed_arguments_ids.append(i)
                    candidate_args.append(t)

            self.toggle_simpset("diminish", goal_theory)

            logging.debug("Removed simpset of {}".format(goal_theory))

        except Exception as e:
            raise Exception(f"Error generating fact pool: {e}")

        return allowed_arguments_ids, candidate_args

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
            for i,e in enumerate(f["content"]):
                if i < len(parent_fringe["content"])-1:
                    # parent_goal_id is the id of the removed parent goal
                    if i < parent_goal_id:
                        e["origin_fringe"] = parent_fringe["content"][i]["origin_fringe"]
                    else:
                        e["origin_fringe"] = parent_fringe["content"][i+1]["origin_fringe"]
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
        new_content = content[len(parent_fringe["content"])-1:]
        dependency_table[(revert_assumptions(parent_goal), parent_goal["origin_fringe"])] = (remove_duplicates(tactic), new_content)

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
                script = script + " >- (rpt (pop_assum mp_tac) >> rpt strip_tac >> {})".format(generate_script(new_key, m))
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
    return (data==[])


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
    for i,p in enumerate(fringe):
        text += "{}: {}<br>".format(i,p["plain"])
        
    return text[:-4]
    

def make_tree(history):
    es = []
    for i in history:
        p = i["parent"]
        if p != None: # p can be 0
            es.append(((p, history.index(i)), # edge
                       (i["goal"], i["by_tactic"]))) # label

    return es

    
def draw_tree(history):#, output_graph=False):
    nv = len(history)
    eslb = make_tree(history)
    es = [i[0] for i in eslb]
    g = Graph(nv, es, True)
    
    g.vs["goals"] = [get_text(i["content"]) for i in history]
    g.es["by applying tactic on"] = [i[1] for i in eslb]
    g.es["by applying tactic on"] = ["Step: {}<br>Target: {}<br>Tactic: {}".format(n, i[1][0], i[1][1]) for (n,i) in enumerate(eslb)]
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
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []    
    for edge in es:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    # fs = [get_text(i["content"]) for i in history]
    
    vlabels = [get_text(i["content"]) for i in history]
    elabels = g.es["by applying tactic on"]
    
    # calculate the middle points for edge labels
    Xel = []
    Yel = []
    for edge in es:
        Xel+=[0.5*(position[edge[0]][0]+position[edge[1]][0])]
        Yel+=[0.5*(2*M-position[edge[0]][1]+2*M-position[edge[1]][1])]
        
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
                                         color='#FF0000',    #'#DB4551',
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
                                             color='#6175c1',    #'#DB4551',
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

#def revert_with_polish(context):
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
       #goal = "@ @ D$min$==> {} {}".format(i, goal)
       goal = "@ @ C$min$ ==> {} {}".format(i, goal)

   return goal 

   

#def gather_encoded_content(history, encoder):
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
