"""Compute embeddings and predictions from a saved holparam checkpoint."""

from __future__ import absolute_import
from __future__ import division

# Import Type Annotations
from __future__ import print_function

import logging
from typing import List
from typing import Optional
from typing import Text

import numpy as np
import torch
from pymongo import MongoClient

from data.utils.graph_data_utils import transform_expr, transform_batch
from experiments.holist.agent import predictions
from data.holist.utils import process_sexp
from data.holist.utils.sexpression_to_graph import sexpression_to_graph
from models.get_model import get_model
from models.holist_models.tactic_predictor import TacticPrecdictor, CombinerNetwork

GOAL_EMB_TYPE = predictions.GOAL_EMB_TYPE
THM_EMB_TYPE = predictions.THM_EMB_TYPE
STATE_ENC_TYPE = predictions.STATE_ENC_TYPE


# handle different data structures
def transfer_to_device(x, device):
    if isinstance(x, tuple):
        return (tuple([transfer_to_device(xi, device) for xi in x]))
    if isinstance(x, list):
        return [transfer_to_device(xi, device) for xi in x]
    else:
        return x.to(device)


def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def recommend_from_scores(scores: List[List[float]], n: int) -> List[List[int]]:
    """Return the index of the top n predicted scores.

    Args:
      scores: A list of tactic probabilities, each of length equal to the number
        of tactics.
      n: The number of recommendations requested.

    Returns:
      A list of the indices with the highest scores.
    """

    def top_idx(scores):
        return np.array(scores).argsort()[::-1][:n]

    return [top_idx(s) for s in scores]


'''

Torch Reimplementation of original TF1 HolparamPredictor

'''


class HolparamPredictor(predictions.Predictions):
    """Compute embeddings and make predictions from a save checkpoint."""

    def __init__(self,
                 ckpt: Text,
                 max_embedding_batch_size: Optional[int] = 512,
                 max_score_batch_size: Optional[int] = 512, config=None) -> None:
        """Restore from the checkpoint into the session."""
        super(HolparamPredictor, self).__init__(
            max_embedding_batch_size=max_embedding_batch_size,
            max_score_batch_size=max_score_batch_size)

        self.config = config
        self.embedding_model_goal = get_model(config.model_config).cuda()
        self.embedding_model_premise = get_model(config.model_config).cuda()

        self.tac_model = TacticPrecdictor(
            embedding_dim=1024,
            num_tactics=41).cuda()

        self.combiner_model = CombinerNetwork(
            embedding_dim=1024,
            num_tactics=41,
            tac_embed_dim=128).cuda()

        # self.load_pretrained_model(ckpt)

        self.embedding_model_goal.eval()
        self.embedding_model_premise.eval()
        self.tac_model.eval()
        self.combiner_model.eval()

        client = MongoClient()

        db = client[self.config.data_config.data_options['db']]
        vocab_col = db[self.config.data_config.data_options['vocab_col']]

        self.filter = self.config.data_config.data_options['filter']

        self.vocab = {k['_id']: k['index'] for k in vocab_col.find()}

    # todo better loading
    def load_pretrained_model(self, ckpt_dir):
        logging.info(f"Loading checkpoint from {ckpt_dir}")
        ckpt = torch.load(ckpt_dir + '.ckpt', map_location={'cuda:1': 'cuda:0'})['state_dict']
        self.embedding_model_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
        self.embedding_model_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))
        self.tac_model.load_state_dict(get_model_dict('tac_model', ckpt))
        self.combiner_model.load_state_dict(get_model_dict('combiner_model', ckpt))

    def to_torch(self, data_dict):
        return transform_expr(data_dict,
                              data_type=self.config.data_config.type,
                              vocab=self.vocab,
                              config=self.config.data_config)

    def _goal_string_for_predictions(self, goals: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(goal) for goal in goals]

    def _thm_string_for_predictions(self, thms: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(thm) for thm in thms]

    def _batch_goal_embedding(self, goals: List[Text]) -> List[GOAL_EMB_TYPE]:
        """From a list of string goals, compute and return their embeddings."""
        # Get the first goal_net collection (second entry may be duplicated to align
        # with negative theorems)
        with torch.no_grad():
            goals = self._goal_string_for_predictions(goals)

            goal_data = [self.to_torch(sexpression_to_graph(t, type=self.config.data_config.type)) for t in goals]

            goal_data = transform_batch(goal_data, config=self.config.data_config)

            goal_data = transfer_to_device(goal_data, 'cuda')

            embeddings = self.embedding_model_goal(goal_data)

            embeddings = embeddings.cpu().numpy()

        return embeddings

    def _batch_thm_embedding(self, thms: List[Text]) -> List[THM_EMB_TYPE]:
        """From a list of string theorems, compute and return their embeddings."""
        # The checkpoint should have exactly one value in this collection.
        with torch.no_grad():
            thms = self._thm_string_for_predictions(thms)

            thms_data = [self.to_torch(sexpression_to_graph(t, type=self.config.data_config.type)) for t in thms]

            thms_data = transform_batch(thms_data, config=self.config.data_config)

            thms_data = transfer_to_device(thms_data, 'cuda')

            embeddings = self.embedding_model_premise(thms_data)
            embeddings = embeddings.cpu().numpy()
        return embeddings

    def thm_embedding(self, thm: Text) -> THM_EMB_TYPE:
        """Given a theorem as a string, compute and return its embedding."""
        # Pack and unpack the thm into a batch of size one.
        [embedding] = self.batch_thm_embedding([thm])
        return embedding

    def proof_state_from_search(self, node) -> predictions.ProofState:
        """Convert from proof_search_tree.ProofSearchNode to ProofState."""
        return predictions.ProofState(goal=str(node.goal.conclusion))

    def proof_state_embedding(
            self, state: predictions.ProofState) -> predictions.EmbProofState:
        return predictions.EmbProofState(goal_emb=self.goal_embedding(state.goal))

    def proof_state_encoding(
            self, state: predictions.EmbProofState) -> STATE_ENC_TYPE:
        return state.goal_emb

    def _batch_tactic_scores(
            self, state_encodings: List[STATE_ENC_TYPE]) -> List[List[float]]:
        """

        Predict tactic probabilities for a batch of goals.

        Args:
          state_encodings: A list of n goal embeddings.

        Returns:
          A list of n tactic probabilities, each of length equal to the number of
            tactics.

        """

        # The checkpoint should have only one value in this collection.
        with torch.no_grad():
            x = torch.from_numpy(np.array(state_encodings))

            # [tactic_scores] = self.tac_model(x).tolist()
            tactic_scores = self.tac_model(x.cuda()).cpu().numpy()

        return tactic_scores

    def _batch_thm_scores(self,
                          state_encodings: List[STATE_ENC_TYPE],
                          thm_embeddings: List[THM_EMB_TYPE],
                          tactic_id: Optional[int] = None) -> List[float]:
        """

        Predict relevance scores for goal, theorem pairs.

        Args:
          state_encodings: A proof state encoding. (effectively goal embedding)
          thm_embeddings: A list of n theorem embeddings. Theorems are paired by
            index with corresponding goals.
          tactic_id: Optionally tactic that the theorem parameters will be used in.

        Returns:
          A list of n floats, representing the pairwise score of each goal, thm.

        """

        del tactic_id  # tactic id not use to predict theorem scores.
        # The checkpoint should have only one value in this collection.
        assert len(state_encodings) == len(thm_embeddings)

        # todo use non tac dependent model

        tactic_id = [0]
        with torch.no_grad():
            scores = self.combiner_model(torch.Tensor(state_encodings).unsqueeze(0).cuda(),
                                         torch.Tensor(thm_embeddings).unsqueeze(0).cuda(),
                                         torch.LongTensor(tactic_id).cuda()).squeeze(0).cpu().numpy()

        return scores


class TacDependentPredictor(HolparamPredictor):
    """Derived class, adds dependence on tactic for computing theorem scores."""

    def __init__(self,
                 ckpt: Text,
                 max_embedding_batch_size: Optional[int] = 512,
                 max_score_batch_size: Optional[int] = 512, config=None) -> None:
        """Restore from the checkpoint into the session."""
        super(TacDependentPredictor, self).__init__(
            ckpt,
            max_embedding_batch_size=max_embedding_batch_size,
            max_score_batch_size=max_score_batch_size, config=config)
        self.selected_tactic = -1

    def _batch_thm_scores(self,
                          state_encodings: List[STATE_ENC_TYPE],
                          thm_embeddings: List[THM_EMB_TYPE],
                          tactic_id: Optional[int] = None) -> List[float]:
        """Predict relevance scores for goal, theorem pairs.

        Args:
          state_encodings: A proof state encoding.
          thm_embeddings: A list of n theorem embeddings. Theorems are paired by
            index with corresponding goals.
          tactic_id: Optionally tactic that the theorem parameters will be used in.

        Returns:
          A list of n floats, representing the pairwise score of each goal, thm.
        """
        # Check that the batch size for states and thms is the same.
        assert len(state_encodings) == len(thm_embeddings)

        # Tile the tactic to the batch size.
        # tactic_ids = np.tile(tactic_id, [len(state_encodings)])

        # The checkpoint should have only one value in this collection.
        with torch.no_grad():
            # todo copy np array
            scores = self.combiner_model(torch.Tensor(state_encodings).unsqueeze(0).cuda(),
                                         torch.Tensor(thm_embeddings).unsqueeze(0).cuda(),
                                         torch.LongTensor([tactic_id]).cuda()).squeeze(0).cpu().numpy()
        return scores
