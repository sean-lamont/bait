import os
import re
import time
from time import sleep

import pexpect
from loguru import logger

from experiments.end_to_end.proof_node import *


class EnvInitError(Exception):
    pass


def get_process(pstring):
    pids = []
    fd = os.popen("ps ax | grep " + pstring + " | grep -v grep")
    for line in fd:
        fields = line.split()
        pid = fields[0]
        pids.append(pid)
    fd.close()
    return pids


def revert_assumptions(context):
    # take a context and return a reverted goal if there are assumptions
    target = context["plain"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in assumptions:
        goal = "(" + i + ")" + " ==> " + "(" + goal + ")"
    return goal


def revert_with_polish(context):
    target = context["polished"]
    assumptions = target["assumptions"]
    goal = target["goal"]
    for i in reversed(assumptions):
        goal = "@ @ C$min$ ==> {} {}".format(i, goal)

    return goal


'''

Environment Wrapper over HOL4. 

'''

# HOLPATH = "/home/sean/Documents/phd/hol/HOL/bin/hol --maxheap=256"

HOLPATH = "environments/HOL4/HOL/bin/hol --maxheap=256"

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

thms_tactic = ["simp", "fs", "metis_tac", "rw"]
thm_tactic = ["irule", "drule"]
term_tactic = ["Induct_on"]
no_arg_tactic = ["strip_tac", "EQ_TAC"]


class HOL4Env:
    def __init__(self, thm, timeout):
        self.timeout = timeout
        self.environment_time = 0

        # maps polished goal string to full representation and corresponding proof node.
        self.node_map = {}

        self.thm, self.database = thm

        self.premises, self.goal_theory = self.retrieve_premises()

    def __enter__(self):
        try:

            self.import_theories = ["probabilityTheory"]
            self.process = pexpect.spawn(HOLPATH, timeout=3)

            self.toggle_simpset(self.goal_theory)

            logger.info("Removed simpset of {}".format(self.goal_theory))

            # experimental feature
            self.process.delaybeforesend = None

            # import theories
            logger.info("Importing theories...")

            self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
            self.process.sendline("val _ = set_trace \"types\" 1;".encode("utf-8"))

            for i in self.import_theories:
                self.process.sendline("load \"{}\";".format(i).encode("utf-8"))
                self.process.sendline("open {};".format(i).encode("utf-8"))
                sleep(5)

            self.process.sendline("use \"helper.sml\";")
            sleep(5)

            logger.info("Configuration done.")
            self.process.expect('\r\n>')
            self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))

            # consumes hol4 head
            self.process.expect('\r\n>')

            # polished goal
            goal = self.thm[1]

            polished_goal = self.get_polish(goal)[0]

            logger.info("Initialization done. Main goal is:\n{}.".format(goal))

            # use polished goal as the state
            init_state = self.thm[0]

        except Exception as e:
            raise EnvInitError(e)

        root = InternalNode(goal=init_state, cumulative_logprob=0.0)

        self.node_map[init_state] = (polished_goal, root)

        return self, root

    def construct_goal(self, goal):
        s = "g " + "`" + goal + "`;"
        return s

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

        pd = eval(polished_subgoals)

        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        data = [{"polished": {"assumptions": e[0][0], "goal": e[0][1]},
                 "plain": {"assumptions": e[1][0], "goal": e[1][1]}}
                for e in zip(pd, [([], raw_goal)])]

        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.close(force=True)

    def toggle_simpset(self, theory):
        reset_cmd = "BasicProvers.recreate_sset_at_parentage (parents \"{}\");".format(theory)
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.sendline(reset_cmd.encode("utf-8"))
        self.process.sendline("val _ = HOL_Interactive.toggle_quietdec();".encode("utf-8"))
        self.process.expect('\r\n>')

    def retrieve_premises(self):
        allowed_theories = list(set(re.findall(r'C\$(\w+)\$ ', self.thm[0])))
        goal_theory = self.database[self.thm[0]][0]

        try:
            candidate_args = []
            for i, t in enumerate(self.database):
                theory_allowed = self.database[t][0] in allowed_theories
                diff_theory = self.database[t][0] != goal_theory
                prev_theory = int(self.database[t][3]) < int(self.database[self.thm[0]][3])
                if theory_allowed and (diff_theory or prev_theory):
                    # allowed_arguments_ids.append(i)
                    candidate_args.append(t)

        except Exception as e:
            raise Exception(f"Error generating fact pool: {e}")

        # return allowed_arguments_ids, candidate_args
        return candidate_args, goal_theory

    def construct_tactic(self, tac, limited_time=True):
        if limited_time:
            s = "e " + "(" + tac + ");"
        else:
            s = "unlimited_e " + "(" + tac + ");"
        return s

    def query(self, raw_goal, tac, limited_time=True):
        self.handling = raw_goal
        self.using = tac

        goal = self.construct_goal(raw_goal)
        self.process.sendline(goal.encode("utf-8"))
        self.process.expect("\r\n>")

        tactic = self.construct_tactic(tac, limited_time)
        self.process.sendline(tactic.encode("utf-8"))

        try:
            i = self.process.expect(
                ["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof", "Exception",
                 "error"])
        except:
            logger.debug("Exception: {} to {} to be debugged".format(tac, raw_goal))
            i = -1

        if i == -1:
            data = "unexpected"
            return data

        # workaround
        while i == 0:
            # skip the proof translation error and read the Exception
            i = self.process.expect(
                ["metis: proof translation error", "Initial goal proved", ": proof", ":\r\n +proof", "Exception",
                 "error"])

            logger.debug("i is {}".format(i))

        if i == 2 or i == 3:
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))

            try:
                self.process.expect("val it =")
            except:
                logger.debug("Exception: {} to {} returned no goals".format(tac, raw_goal))
                return "exception"

            # this (:\r\n) doesn't seem robust
            self.process.expect([": goal list", ":\r\n +goal list"])
            raw = self.process.before.decode("utf-8")

            subgoals = re.sub("“|”", "\"", raw)
            subgoals = re.sub("\r\n +", " ", subgoals)

            # get Polished version
            self.process.expect("\r\n>")
            self.process.sendline("val _ = set_term_printer (HOLPP.add_string o pt);".encode("utf-8"))
            self.process.expect("\r\n>")
            self.process.sendline("top_goals();".encode("utf-8"))
            self.process.expect("val it =")
            self.process.expect([": goal list", ":\r\n +goal list"])

            polished_raw = self.process.before.decode("utf-8")

            polished_subgoals = re.sub("“|”", "\"", polished_raw)
            polished_subgoals = re.sub("\r\n +", " ", polished_subgoals)

            # escape colored characters
            polished_subgoals = ansi_escape.sub('', polished_subgoals)
            subgoals = ansi_escape.sub('', subgoals)

            pd = eval(polished_subgoals)
            d = eval(subgoals)

            data = zip(pd, d)
            data = [{"polished": {"assumptions": e[0][0], "goal": e[0][1]},
                     "plain": {"assumptions": e[1][0], "goal": e[1][1]}}
                    for e in data]

        elif i == 1:
            data = []
        elif i == 4:
            j = self.process.expect(["Time", pexpect.TIMEOUT], timeout=0.01)

            if j == 0:
                data = "timeout"
            else:
                data = "exception"
        else:
            logger.debug("Exception: {} to {}.".format(tac, raw_goal))
            data = "exception"

        # clear stack and consume the remaining stdout
        self.process.expect("\r\n>")
        self.process.sendline("drop();".encode("utf-8"))
        self.process.expect("\r\n>")
        self.process.sendline("val _ = set_term_printer default_pt;".encode("utf-8"))
        self.process.expect("\r\n>")

        return data

    def get_names(self, exps):
        # look up the names of exps
        names = []
        for e in exps:
            theorem_name = self.database[e][1]
            theory_name = self.database[e][0]
            full_name = theory_name + "Theory." + theorem_name
            names.append(full_name)
        return names

    def assemble_tactic(self, tac):
        # assume args are split by comma

        tac = tac.split("<arg>")

        if len(tac) > 1:
            tac, args = tac[0], tac[1:]

            # args is a list of strings
            if tac in thms_tactic:
                names = self.get_names(args)
                action = tac + re.sub("'", "", str(names))
            # args just a single theorem
            else:
                names = self.get_names(args)
                if names:
                    action = tac + " " + names[0]
                else:
                    # this will raise an error in HOL4
                    action = tac
        else:
            # term tactics will be already assembled
            # no_arg_tactic are handled as is
            action = tac[0]

        return action

    def run_tactic(self, node, tactic):
        tactic, tac_logprob = tactic

        tactic = self.assemble_tactic(tactic)

        t0 = time.monotonic()

        node, goal_logprob = node

        pre_target = self.node_map[node.goal][0]

        target = pre_target["plain"]

        if target["assumptions"]:
            # there are assumptions
            goal = revert_assumptions(pre_target)
            logger.info(f'running goal with tactic {goal, tactic}')
            response = self.query(goal, "rpt strip_tac >> " + tactic)
        else:
            # no assumptions
            goal = target["goal"]
            logger.info(f'running goal with tactic {goal, tactic}')
            response = self.query(goal, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        # process response here

        result_node = []

        logger.info(f'tactic response: {response}')

        if response == "unexpected" or response == "exception" or response == "timeout":
            response = EnvironmentError(message=response)
            result_node = ErrorNode(response)
            result = [result_node]

        else:
            # progress has been made
            if [pre_target] != response:
                # solved
                if response == []:
                    result_node = ProofFinishedNode(GoalFinished())
                    result = [result_node]
                else:
                    result = []
                    for i, context in enumerate(response):
                        full_goal = revert_with_polish(context)

                        if full_goal in node.ancestors:
                            response = TreeError('Tactic Creates cycle')
                            result_node = ErrorNode(response)
                            result = [result_node]
                            break
                        if full_goal in self.node_map:
                            context, result_node = self.node_map[full_goal]
                        else:
                            result_node = InternalNode(
                                goal=full_goal,
                                cumulative_logprob=tac_logprob + node.cumulative_logprob,
                                depth=node.depth + 1
                            )

                            self.node_map[full_goal] = (context, result_node)

                        # todo add below to search processing?
                        # This will add the parent context (any goals required to prove the parent)
                        # as well as other siblings from the current result.
                        sib_context = {revert_with_polish(goal_) for goal_ in response if goal_ != context}
                        if node.context:
                            cur_context = [ctx | sib_context for ctx in node.context]
                        else:
                            cur_context = [sib_context]

                        result_node.add_context(cur_context)

                        # Add ancestors for detecting cycles
                        result_node.add_ancestors(node.ancestors | {node.goal})

                        result.append(result_node)

            # nothing changed
            else:
                response = TreeError('No Progress')
                result_node = ErrorNode(response)
                result = [result_node]

        # self-loop sanity check (should never happen)
        if result_node == node:
            logger.error(f'Self loop found')
            response = TreeError('Self-loop')
            result_node = ErrorNode(response)
            result = [result_node]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result, tac_logprob=tac_logprob, goal_logprob=goal_logprob,
                    time=elapsed)

        if node.out_edges:
            node.out_edges = node.out_edges + [edge]
        else:
            node.out_edges = [edge]

        for result_node in result:
            if isinstance(result_node, InternalNode):
                result_node.in_edges.append(edge)

        return edge
