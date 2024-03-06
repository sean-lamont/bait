"""

PyTorch Lightning module for training HOList models from labelled data

"""

import logging
import traceback
import warnings
import einops

from experiments.end_to_end.common import load_checkpoint

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
import torch


def auroc(pos, neg):
    return torch.mean(torch.log(1 + torch.exp(-1 * (pos - neg))))


ce_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCEWithLogitsLoss()


# todo live evaluation, combine with old HOList train module (since there are no changes)


class HOListTraining_(pl.LightningModule):
    def __init__(self,
                 goal_embedding_module,
                 premise_embedding_module,
                 tac_model,
                 combiner_model,
                 batch_size=16,
                 lr=1e-4):
        super().__init__()

        torch.set_float32_matmul_precision('medium')
        self.embedding_model_goal = goal_embedding_module
        self.embedding_model_premise = premise_embedding_module
        self.tac_model = tac_model
        self.combiner_model = combiner_model
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = batch_size

    '''

    Calculate loss function as defined in original implementation paper. 

    neg_premise_scores are scores for negative premises, and extra_neg_premises are additional negatives sampled from the batch.
    They are weighted differently to favour negatives from the same goal

    '''

    def loss_func(self, tac_pred, true_tac, pos_premise_scores, neg_premise_scores, extra_neg_premise_scores,
                  tac_weight=1,
                  pairwise_weight=0.2,
                  auroc_weight=4,
                  same_goal_weight=2):

        tac_loss = ce_loss(tac_pred, true_tac)
        pairwise_loss_positives = bce_loss(pos_premise_scores.squeeze(1),
                                           torch.ones(pos_premise_scores.shape[0]).to(self.device))

        pairwise_loss_main_negatives = bce_loss(neg_premise_scores.flatten(0, 1), torch.zeros(
            neg_premise_scores.shape[0] * neg_premise_scores.shape[1]).to(self.device))

        pairwise_loss_extra_negatives = bce_loss(extra_neg_premise_scores.flatten(0, 1), torch.zeros(
            extra_neg_premise_scores.shape[0] * extra_neg_premise_scores.shape[1]).to(self.device))

        pos_premise_scores_main_negatives = einops.repeat(pos_premise_scores, 'b 1 -> b k',
                                                          k=neg_premise_scores.shape[-1])

        auroc_loss_main_negatives = auroc(pos_premise_scores_main_negatives, neg_premise_scores)

        pos_premise_scores_extra_negatives = einops.repeat(pos_premise_scores, 'b 1 -> b k',
                                                           k=extra_neg_premise_scores.shape[-1])
        auroc_loss_extra_negatives = auroc(pos_premise_scores_extra_negatives, extra_neg_premise_scores)

        final_loss = tac_weight * tac_loss \
                     + pairwise_weight * (
                             pairwise_loss_positives + pairwise_loss_extra_negatives + pairwise_loss_main_negatives) \
                     + auroc_weight * ((same_goal_weight * auroc_loss_main_negatives) + auroc_loss_extra_negatives)

        return final_loss

    def val_func(self, tac_pred, true_tac, pos_premise_scores, neg_premise_scores, extra_neg_premise_scores):
        tac_acc = torch.sum(torch.argmax(tac_pred, dim=1) == true_tac) / tac_pred.shape[0]
        topk_preds = torch.topk(tac_pred, k=5, dim=1).indices
        topk_acc = sum([1 if torch.isin(true_tac[i], topk_preds[i]) else 0 for i in range(tac_pred.shape[0])]) / \
                   tac_pred.shape[0]

        neg_premise_scores = torch.cat([neg_premise_scores, extra_neg_premise_scores], dim=1)
        pos_premise_scores_dupe = einops.repeat(pos_premise_scores, 'b 1 -> b k',
                                                k=neg_premise_scores.shape[-1])

        rel_param_acc = torch.sum(pos_premise_scores_dupe > neg_premise_scores) / (
                neg_premise_scores.shape[0] * neg_premise_scores.shape[1])

        pos_acc = torch.sum(torch.sigmoid(pos_premise_scores) > 0.5) / pos_premise_scores.shape[0]
        neg_acc = torch.sum(torch.sigmoid(neg_premise_scores) < 0.5) / (
                neg_premise_scores.shape[0] * neg_premise_scores.shape[1])

        return tac_acc, rel_param_acc, pos_acc, neg_acc, topk_acc

    def forward(self, goals, pos_thms, neg_thms, true_tacs):
        goals = self.embedding_model_goal(goals).unsqueeze(1)
        pos_thms = self.embedding_model_premise(pos_thms).unsqueeze(1)
        neg_thms = torch.stack([self.embedding_model_premise(neg_thm) for neg_thm in neg_thms], dim=0)

        # construct extra_neg_thms after embedding, since embedding is the most computationally expensive step
        extra_neg_thms = torch.stack([torch.cat([torch.cat([pos_thms[:i], pos_thms[(i + 1):]], dim=0),
                                                 torch.cat([neg_thms[:i], neg_thms[(i + 1):]], dim=0)],
                                                dim=1).flatten(0, 1)
                                      for i in range(len(goals))], dim=0)

        tac_preds = self.tac_model(goals)

        pos_scores = self.combiner_model(goals, pos_thms, true_tacs)
        neg_scores = self.combiner_model(einops.repeat(goals, 'b 1 d -> b k d', k=neg_thms.shape[1]), neg_thms,
                                         true_tacs)

        extra_neg_scores = self.combiner_model(einops.repeat(goals, 'b 1 d -> b k d', k=extra_neg_thms.shape[1]),
                                               extra_neg_thms, true_tacs)

        return tac_preds.flatten(1, 2), pos_scores.flatten(1, 2), neg_scores.flatten(1, 2), extra_neg_scores.flatten(1,
                                                                                                                     2)

    def training_step(self, batch, batch_idx):
        goals, true_tacs, pos_thms, neg_thms = batch
        try:
            tac_preds, pos_scores, neg_scores, extra_neg_scores = self(goals, pos_thms, neg_thms, true_tacs)
        except Exception as e:
            logging.debug(f"Error in forward: {e}")
            traceback.print_exc()
            return

        loss = self.loss_func(tac_preds, true_tacs,
                              pos_scores, neg_scores,
                              extra_neg_scores)

        if not torch.isfinite(loss):
            logging.debug(f"Loss error: {loss}")
            return

        self.log("loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        goals, true_tacs, pos_thms, neg_thms = batch
        try:
            tac_preds, pos_scores, neg_scores, extra_neg_scores = self(goals, pos_thms, neg_thms, true_tacs)
        except Exception as e:
            logging.debug(f"Error in forward: {e}")
            return

        # get accuracy wrt true
        tac_acc, rel_param_acc, pos_acc, neg_acc, topk_acc = self.val_func(tac_preds, true_tacs, pos_scores, neg_scores,
                                                                           extra_neg_scores)

        self.log_dict({'tac_acc': tac_acc, 'rel_param_acc': rel_param_acc, 'pos_acc': pos_acc, 'neg_acc': neg_acc,
                       'topk_acc': topk_acc},
                      batch_size=self.batch_size, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            logging.debug(f"Error in backward: {e}")

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)
