import logging
import time
import traceback

from experiments.hol4_tactic_zero.rl.old.agent_utils import *
from experiments.hol4_tactic_zero.rl.old.rl_data_module import *
import lightning.pytorch as pl
import torch.optim
from torch.distributions import Categorical
from environments.hol4.new_env import *
import warnings

from utils.viz_net_torch import make_dot

warnings.filterwarnings('ignore')

"""

 Torch Lightning TacticZero Loop:
 
"""


class TacticZeroLoop(pl.LightningModule):
    def __init__(self,
                 context_net,
                 tac_net,
                 arg_net,
                 term_net,
                 induct_net,
                 encoder_premise,
                 encoder_goal,
                 config,
                 graph_db,
                 token_enc,
                 tactics,
                 reverse_database,
                 ):

        super().__init__()

        self.context_net = context_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal
        self.proven = []
        self.cumulative_proven = []
        self.replayed = 0
        self.val_proved = []

        self.graph_db = graph_db
        self.token_enc = token_enc
        self.reverse_database = reverse_database

        self.thms_tactic = tactics['thms_tactic']
        self.thm_tactic = tactics['thm_tactic']
        self.term_tactic = tactics['term_tactic']
        self.no_arg_tactic = tactics['no_arg_tactic']

        self.tactic_pool = tactics['tactic_pool']

        self.config = config
        self.dir = self.config['dir_path']
        os.makedirs(self.dir, exist_ok=True)

        logging.basicConfig(filename=self.dir + 'log', level=logging.DEBUG)

        # self.cur_steps = 1

        # todo replay DB + log probs
        if os.path.exists(self.dir + 'replays'):
            self.replays = torch.load(self.config['replay_dir'])
            self.replay_dir = self.config['replay_dir']
            # add to cumulative list for accurate logging
            self.cumulative_proven = [None for _ in self.replays]
        else:
            print(f"Creating new replay dir {self.dir + 'replays'}")
            self.replays = {}
            self.replay_dir = self.dir + 'replays'

    def forward(self, batch, train_mode=True):
        goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
        # print (f"goal: {goal} \n")

        # encodings = [self.encoder_premise(fact_batch) for fact_batch in allowed_fact_batch]
        # encodings = [self.encoder_premise(fact_batch) for fact_batch in allowed_fact_batch]
        # encoded_fact_pool = torch.cat(encodings, dim=0)


        encoded_fact_pool = self.encoder_premise(allowed_fact_batch)

        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0
        # start_t = time.time()

        # todo stats per goal? save to experiment directory?

        # todo 1 - eps replay? I.e run replay only with prob 1 - eps

        # todo learnable step count? I.e. Model predicts num steps up to max, then if it proves within step gets a large reward, else punished?

        # dynamic step count: Start at 1, if goal has been proven max_steps is fixed value above that (e.g. 5), if not have counter of current max which grows over time

        # if env.goal in self.replays:
        #     max_steps = self.replays[env.goal][0] + 5
        # else:

        max_steps = self.config['max_steps']  # self.cur_steps


        for t in range(max_steps):
            target_representation, target_goal, fringe, fringe_prob = select_goal_fringe(history=env.history,
                                                                                              encoder_goal=self.encoder_goal,
                                                                                              graph_db=self.graph_db,
                                                                                              token_enc=self.token_enc,
                                                                                              context_net=self.context_net,
                                                                                              device=self.device,
                                                                                              data_type=self.config[
                                                                                                  'data_type'],
                                                                                              )

            # print (target_representation.requires_grad, fringe.requires_grad, fringe_prob.requires_grad)

            fringe_pool.append(fringe_prob)
            tac, tac_prob = get_tac(tac_input=target_representation,
                                    tac_net=self.tac_net,
                                    device=self.device)

            tac_pool.append(tac_prob)

            if self.tactic_pool[tac] in self.no_arg_tactic:
                tactic = self.tactic_pool[tac]
                arg_probs = [torch.tensor(0)]

            elif self.tactic_pool[tac] == "Induct_on":
                tactic, arg_probs = get_term_tac(target_goal=target_goal,
                                                 target_representation=target_representation,
                                                 tac=tac,
                                                 term_net=self.term_net,
                                                 induct_net=self.induct_net,
                                                 device=self.device,
                                                 token_enc=self.token_enc)

            else:
                tactic, arg_probs = get_arg_tac(target_representation=target_representation,
                                                num_args=len(allowed_arguments_ids),
                                                encoded_fact_pool=encoded_fact_pool,
                                                tac=tac,
                                                candidate_args=candidate_args,
                                                env=env,
                                                device=self.device,
                                                arg_net=self.arg_net,
                                                arg_len=self.config['arg_len'],
                                                reverse_database=self.reverse_database)

            arg_pool.append(arg_probs)
            action = (fringe.detach(), 0, tactic)

            try:
                reward, done = env.step(action)
            except Exception as e:
                # todo reset or continue?
                env = HolEnv("T")
                return ("Step error", action)
                # reward = -1
                # done = False


            if done:
                # if not train_mode:
                #     break

                self.proven.append([env.polished_goal[0], t + 1])

                if env.goal in self.replays.keys():
                    if steps < self.replays[env.goal][0]:
                        self.replays[env.goal] = (steps, env.history)
                else:
                    self.cumulative_proven.append([env.polished_goal[0]])
                    if env.history is not None:
                        self.replays[env.goal] = (steps, env.history)
                    else:
                        print("History is none.")
                        print(env.history)
                        print(env)

                reward_pool.append(reward)
                steps += 1
                break

            if t == max_steps - 1:
                reward = -5
                # print (steps, len(reward_pool))
                reward_pool.append(reward)

                # if not train_mode:
                #     break

                if env.goal in self.replays:
                    return self.run_replay(allowed_arguments_ids, candidate_args, env, encoded_fact_pool)

            reward_pool.append(reward)
            steps += 1
        # print (f"outcome: {reward_pool, fringe_pool, tac_pool, steps, done}")

        # if train_mode:
        #     g = make_dot(tac_pool[0])
        #     g.view()
        #     time.sleep(100)

        return reward_pool, fringe_pool, arg_pool, tac_pool, steps, done

    def run_replay(self, allowed_arguments_ids, candidate_args, env, encoded_fact_pool):

        # todo graph replay:
        # reps = self.replays[env.goal]
        # rep_lens = [len(rep[0]) for rep in reps]
        # min_rep = reps[rep_lens.index(min(rep_lens))]
        # known_history, known_action_history, reward_history, _ = min_rep

        reward_pool = []
        fringe_pool = []
        arg_pool = []
        tac_pool = []
        steps = 0

        known_history = self.replays[env.goal][1]


        for t in range(len(known_history) - 1):
            true_resulting_fringe = known_history[t + 1]
            true_fringe = torch.tensor([true_resulting_fringe["parent"]], device=self.device)  # .to(self.device)

            target_representation, target_goal, fringe, fringe_prob = select_goal_fringe(
                history=known_history[:t + 1],
                encoder_goal=self.encoder_goal,
                graph_db=self.graph_db,
                token_enc=self.token_enc,
                context_net=self.context_net,
                device=self.device,
                replay_fringe=true_fringe,
                data_type=self.config['data_type'],
                )

            fringe_pool.append(fringe_prob)
            tac_probs = self.tac_net(target_representation)
            tac_m = Categorical(tac_probs)

            true_tactic_text = true_resulting_fringe["by_tactic"]
            true_tac_text, true_args_text = get_replay_tac(true_tactic_text)

            true_tac = torch.tensor([self.tactic_pool.index(true_tac_text)], device=self.device)  # .to(self.device)
            tac_pool.append(tac_m.log_prob(true_tac))

            assert self.tactic_pool[true_tac.detach()] == true_tac_text

            if self.tactic_pool[true_tac] in self.no_arg_tactic:
                arg_probs = [torch.tensor(0)]
                # ??
                arg_pool.append(arg_probs)

            elif self.tactic_pool[true_tac] == "Induct_on":
                _, arg_probs = get_term_tac(target_goal=target_goal,
                                            target_representation=target_representation,
                                            tac=true_tac,
                                            term_net=self.term_net,
                                            induct_net=self.induct_net,
                                            device=self.device,
                                            token_enc=self.token_enc,
                                            replay_term=true_args_text)
            else:
                _, arg_probs = get_arg_tac(target_representation=target_representation,
                                           num_args=len(allowed_arguments_ids),
                                           encoded_fact_pool=encoded_fact_pool,
                                           tac=true_tac,
                                           candidate_args=candidate_args,
                                           env=env,
                                           device=self.device,
                                           arg_net=self.arg_net,
                                           arg_len=self.config['arg_len'],
                                           reverse_database=self.reverse_database,
                                           replay_arg=true_args_text)

            arg_pool.append(arg_probs)
            reward = true_resulting_fringe["reward"]
            reward_pool.append(reward)
            steps += 1

        # print (f"Replay took {steps}")
        return reward_pool, fringe_pool, arg_pool, tac_pool, steps, False

    def save_replays(self):
        torch.save(self.replays, self.replay_dir)

    def training_step(self, batch, batch_idx):
        if batch is None:
            logging.debug("Error in batch")
            return
        try:
            out = self(batch)

            if len(out) == 2:
                logging.debug(f"Error in run: {out}")
                return

            reward_pool, fringe_pool, arg_pool, tac_pool, steps, done = out
            loss = self.update_params(reward_pool, fringe_pool, arg_pool, tac_pool, steps)

            if type(loss) != torch.Tensor:
                logging.debug(f"Error in loss: {loss}")
                return
            return loss

        except Exception as e:
            # print ("error!!!")
            # traceback.print_exc()
            logging.debug(f"Error in training: {e}")
            return

    # def validation_step(self, batch, batch_idx):
    #     if batch is None:
    #         logging.debug("Error in batch")
    #         return
    #     try:
    #         out = self(batch, train_mode=False)
    #         if len(out) == 2:
    #             logging.debug(f"Error in run: {out}")
    #             return
    #         reward_pool, fringe_pool, arg_pool, tac_pool, steps, done = out
    #
    #         if done:
    #             # todo move train version to train_step with batch_idx
    #             self.val_proved.append(batch_idx)
    #         return
    #
    #     except:
    #         logging.debug(f"Error in training: {traceback.print_exc()}")
    #         return

    def on_train_epoch_end(self):
        self.log_dict({"epoch_proven": len(self.proven),
                       "cumulative_proven": len(self.cumulative_proven)},
                      prog_bar=True)

        # todo logging goals, steps etc. proven...
        self.proven = []
        self.save_replays()

    # def on_validation_epoch_end(self):
    #     self.log('val_proven', len(self.val_proved), prog_bar=True)
    #     self.val_proved = []

    def update_params(self, reward_pool, fringe_pool, arg_pool, tac_pool, steps):
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.config['gamma'] + reward_pool[i]
                reward_pool[i] = running_add
        total_loss = 0
        for i in range(steps):
            reward = reward_pool[i]
            fringe_loss = -fringe_pool[i] * (reward)
            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            tac_loss = -tac_pool[i] * (reward)
            loss = fringe_loss + tac_loss + arg_loss
            total_loss += loss


        # g = make_dot(fringe_loss)
        # g.view()
        # time.sleep(100)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config['lr'])
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
            # print ("Gradients: \n\n")
            # for n,p in self.named_parameters():
            #     print (n, p.grad)
            # exit()
        except Exception as e:
            logging.debug(f"Error in backward {e}")
