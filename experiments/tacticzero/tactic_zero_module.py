import logging
from time import sleep
import traceback
import warnings
from abc import abstractmethod

import lightning.pytorch as pl
import torch.optim

from utils.viz_net_torch import make_dot

warnings.filterwarnings('ignore')

"""

Generic TacticZero Abstract Class
 
"""


class TacticZeroLoop(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proven = []
        self.cumulative_proven = []
        self.val_proved = []

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def save_replays(self):
        pass

    @abstractmethod
    def run_replay(self):
        pass

    def training_step(self, batch, batch_idx):
        if batch is None:
            logging.warning("Error in batch")
            return
        try:
            out = self(batch)
            if len(out) != 6:
                logging.warning(f"Error in run: {out}")
                return

            reward_pool, goal_pool, arg_pool, tac_pool, steps, done = out
            loss = self.update_params(reward_pool, goal_pool, arg_pool, tac_pool, steps)

            if type(loss) != torch.Tensor:
                logging.warning(f"Error in loss: {loss}")
                traceback.print_exc()
                return

            return loss

        except Exception as e:
            logging.warning(f"Error in training: {traceback.print_exc()}")
            return

    def validation_step(self, batch, batch_idx):
        if batch is None:
            logging.warning("Error in batch")
            return
        try:
            out = self(batch, train_mode=False)
            if len(out) != 6:
                logging.warning(f"Error in run: {out}")
                return
            reward_pool, goal_pool, arg_pool, tac_pool, steps, done = out

            if done:
                self.val_proved.append(batch_idx)
            return

        except:
            logging.warning(f"Error in training: {traceback.print_exc()}")
            return

    def on_train_epoch_end(self):
        self.log_dict({"epoch_proven": len(self.proven),
                       "cumulative_proven": len(self.cumulative_proven)},
                      prog_bar=True)

        self.proven = []
        self.save_replays()

    def on_validation_epoch_end(self):
        self.log('val_proven', len(self.val_proved), prog_bar=True)
        self.val_proved = []


    def update_params(self, reward_pool, goal_pool, arg_pool, tac_pool, steps):
        running_add = 0

        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.config.gamma + reward_pool[i]
                reward_pool[i] = running_add

        total_loss = 0

        for i in range(steps):
            reward = reward_pool[i]
            goal_loss = -goal_pool[i] * (reward)


            arg_loss = -torch.sum(torch.stack(arg_pool[i])) * (reward)
            tac_loss = -tac_pool[i] * (reward)
            loss = goal_loss + tac_loss + arg_loss
            total_loss += loss

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config.optimiser_config.learning_rate)
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            logging.warning(f"Error in backward {e}")
