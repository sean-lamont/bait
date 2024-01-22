"""Action Generator API.

From information about theorem prover's state, generate a set of possible
actions to take in the prover.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from typing import List, Tuple, Text

import numpy as np
import scipy
from loguru import logger

from data.HOList.utils import normalization_lib, process_sexp, theorem_fingerprint
from environments.HOList.proof_assistant import proof_assistant_pb2
from experiments.HOList import deephol_pb2
from models.end_to_end.tactic_models.holist_model import predictions

EPSILON = 1e-12
MAX_CLOSEST = 100
MIN_SCORED_PARAMETERS = 1
WORD_WEIGHTS_NOISE_SCALE = 1.0


def _theorem_string_for_similarity_scorer(thm) -> Text:
    return process_sexp.process_sexp(str(thm.conclusion))


class SimilarityScorer(object):
    """SimilarityScorer."""

    def __init__(self, theorem_database):
        # We assume theorem database is normalized (so can use tokens directly).
        self.theorem_database = theorem_database
        self.num_words = 0
        self.word_to_index = {}
        self.vocab = []
        self.freq = []

        # build vocab, freq
        for theorem in self.theorem_database.theorems:
            if theorem.training_split != proof_assistant_pb2.Theorem.TRAINING:
                continue
            words = _theorem_string_for_similarity_scorer(theorem).split()
            for word in words:
                if word in self.word_to_index:
                    index = self.word_to_index[word]
                else:
                    index = self.num_words
                    self.num_words += 1
                    self.word_to_index[word] = index
                    self.vocab.append(word)
                    self.freq.append(0)
                self.freq[index] += 1
        freq_sum = sum(self.freq)
        self.inv_freq = np.array([1.0 / float(f) for f in self.freq])

        logger.info(f'Vocab size: {self.num_words}')
        logger.info(f'Frequency sum: {freq_sum}')
        self.reset_word_weights()

    def reset_word_weights(self):
        """Reset word weights, and recompute premise_vectors."""
        logger.info('Resetting word weights')

        self.word_weights = np.multiply(
            self.inv_freq,
            np.absolute(
                np.random.normal(
                    loc=1.0, scale=WORD_WEIGHTS_NOISE_SCALE, size=self.num_words)))

        self.premise_vectors = np.array([
            self.vectorize(_theorem_string_for_similarity_scorer(theorem))
            for theorem in self.theorem_database.theorems
        ])

    def vectorize(self, sentence: Text):
        v = np.zeros(self.num_words)
        for word in sentence.split():
            # TODO(kbk): Consider counting words not in index.
            if word in self.word_to_index:
                index = self.word_to_index[word]
                v[index] += self.word_weights[index]
        return v


def _compute_parameter_string(types, pass_no_arguments: bool,
                              thm_ranked: List[Tuple[float, Text]]
                              ) -> List[Text]:
    """Computes appropriate parameters from a ranked list based on tactic type.

    Args:
      types: Expected type of computed parameters (e.g. thm, list of thms, etc),
        of type List[deephol_pb2.Tactic.ParameterType].
      pass_no_arguments: Pass no parameters to the tactic.
      thm_ranked: ranked theorem parameters.

    Returns:
      A list of string-represented parameter candidates.
    Raises:
      ValueError: if appropriate parameter candidates cannot be generated.
    """
    if not types:
        return ['']
    if not thm_ranked:
        raise ValueError('Theorem parameters are required.')
    if types == [deephol_pb2.Tactic.THEOREM]:
        return [' %s' % thm_ranked[0][1]]

    if types == [deephol_pb2.Tactic.THEOREM_LIST]:
        ret = []
        if not thm_ranked:
            ret.append(' [ ]')
            return ret

        # If predictor also suggests passing no arguments to the tactic, then
        # additionally return an empty list as a parameter string.
        if pass_no_arguments:
            ret.append(' [ ]')

        best_thms = [t for _, t in thm_ranked]
        ret.append(' [ %s ]' % ' ; '.join(best_thms))
        return ret
    raise ValueError('Unsupported tactic parameter types %s' % str(types))


class ActionGenerator(object):
    """Generates candidate actions given the theorem prover's current state."""

    def __init__(
            self,
            theorem_database,
            tactics,
            predictor,
            options,
            model_architecture,
            emb_store=None,
            max_tactics=41
    ):

        self.max_tactics = max_tactics
        self.theorem_database = theorem_database
        self.tactics = tactics
        self.predictor = predictor
        self.options = options
        self.model_architecture = model_architecture
        self.embedding_store = emb_store

        self.thm_names = [
            theorem_fingerprint.ToTacticArgument(thm)
            for thm in theorem_database.theorems
        ]

        self.thm_index_by_fingerprint = {
            theorem_fingerprint.Fingerprint(thm): i
            for (i, thm) in enumerate(theorem_database.theorems)
        }

        # similarity scorer only used for bag of words closest
        self.similarity_scorer = SimilarityScorer(self.theorem_database)

    def _get_theorem_scores(self, proof_state_enc, thm_number: int,
                            tactic_id: int):
        """Get the scores of all the theorems before the given theorem index.

        This functions scores all preceding theorems in the list of theorems, by
        computing all pairwise scores with the given proof state encoding.

        Args:
           proof_state_enc: A numpy vector of the proof state encoding.
           thm_number: Index of the theorem in the theorem database.
           tactic_id: For tactic dependent prediction, provide tactic id.

        Returns:
           A numpy vector of theorem scores for all preceding theorems in the
           same order they are present in the theorem database.
        """
        if self.embedding_store:
            return self.embedding_store.get_thm_scores_for_preceding_thms(
                proof_state_enc, thm_number, tactic_id)

        relevant_thms = self.theorem_database.theorems[:thm_number]

        if relevant_thms:
            thms_emb = self.predictor.batch_thm_embedding([
                normalization_lib.normalize(thm).conclusion for thm in relevant_thms
            ])
        else:
            thms_emb = np.empty([0])
        if len(thms_emb):
            thm_scores = self.predictor.batch_thm_scores(proof_state_enc, thms_emb,
                                                         tactic_id)
        else:
            thm_scores = []

        return thm_scores

    def _compute_tactic_scores(self, proof_state_encoded):
        if self.options.random_tactic_probability > np.random.random():
            return np.random.random([len(self.tactics)])
        return self.predictor.batch_tactic_scores([proof_state_encoded])[0]

    def compute_closest(self, goal, thm_number):
        if not (hasattr(self.options, 'num_similar_parameters') and
                self.options.num_similar_parameters.max_value > 0):
            return None
        if self.options.bag_of_words_similar:
            return self.compute_bag_of_words_closest(goal, thm_number)

        return self.compute_network_based_closest(goal, thm_number)

    def compute_bag_of_words_closest(self, goal, thm_number):
        self.similarity_scorer.reset_word_weights()

        goal_vector = self.similarity_scorer.vectorize(
            _theorem_string_for_similarity_scorer(goal))

        distance_scores = scipy.spatial.distance.cdist(
            self.similarity_scorer.premise_vectors[:thm_number],
            goal_vector.reshape(1, -1), 'cosine').reshape(-1).tolist()
        ranked_closest = sorted(zip(distance_scores, self.thm_names))

        return ranked_closest[:self.options.max_theorem_parameters]

    def compute_network_based_closest(self, goal, thm_number):
        """Compute closest based on premise embeddings."""

        # goal_embedding_as_thm = self.predictor.thm_embedding(
        #     normalization_lib.normalize(goal).conclusion)

        goal_embedding_as_thm = self.predictor.thm_embedding(goal)

        premise_embeddings = (
            self.embedding_store.get_embeddings_for_preceding_thms(thm_number))

        # distance_score each is in [0,2]
        distance_scores = scipy.spatial.distance.cdist(
            premise_embeddings, goal_embedding_as_thm.reshape(1, -1),
            'cosine').reshape(-1).tolist()
        ranked_closest = sorted(zip(distance_scores, self.thm_names))
        ranked_closest = ranked_closest[:MAX_CLOSEST]
        logger.info(
            'Cosine closest in premise embedding space:\n%s', '\n'.join(
                ['%s: %.6f' % (name, score) for score, name in ranked_closest]))
        # add some noise to top few and rerank
        noise = np.random.normal(scale=0.2, size=MAX_CLOSEST)
        ranked_closest = [(score + noise[i], name)
                          for i, (score, name) in enumerate(ranked_closest)]
        ranked_closest = sorted(ranked_closest)
        return ranked_closest[:self.options.max_theorem_parameters]

    def add_similar(self, thm_ranked, ranked_closest):
        """Mix in provided ranked_closest theorems to thm_ranked."""
        if not ranked_closest:
            return thm_ranked[:self.options.max_theorem_parameters]
        num_similar = np.random.random_integers(
            self.options.num_similar_parameters.min_value,
            self.options.num_similar_parameters.max_value)
        num_similar = min(
            num_similar,
            self.options.max_theorem_parameters - MIN_SCORED_PARAMETERS)
        ranked_closest = ranked_closest[:num_similar]

        # remove duplicates
        ranked_closest_names = [name for score, name in ranked_closest]
        thm_ranked = [(score, name)
                      for score, name in thm_ranked
                      if name not in ranked_closest_names]
        return (ranked_closest + thm_ranked)[:self.options.max_theorem_parameters]

    def get_tactics(self, goal, premises):
        max_tactics = self.max_tactics
        premises, thm_number = premises

        # fp = premises.sections[0].before_premise
        # thm_number = self.thm_index_by_fingerprint.get(fp)

        proof_state = predictions.ProofState(goal=goal)

        proof_state_emb = self.predictor.proof_state_embedding(proof_state)

        proof_state_enc = self.predictor.proof_state_encoding(proof_state_emb)

        tactic_scores = self._compute_tactic_scores(proof_state_enc)

        empty_emb = self.predictor.thm_embedding('NO_PARAM')

        empty_emb_batch = np.reshape(empty_emb, [1, empty_emb.shape[0]])

        enumerated_tactics = enumerate(self.tactics)
        if self.options.asm_meson_only:
            enumerated_tactics = [
                v for v in enumerated_tactics if str(v[1].name) == 'ASM_MESON_TAC'
            ]
            assert enumerated_tactics, (
                'action generator option asm_meson_only requires ASM_MESON_TAC.')

            top_tacs = [enumerated_tactics[0][0]]

        else:
            # get top tactics and compute parameters only for these
            top_tacs = np.argpartition(tactic_scores, -max_tactics)[-max_tactics:]

        # compute closest theorems to goal
        ranked_closest = self.compute_closest(goal, thm_number)

        if ranked_closest:
            logger.info(
                'Cosine closest picked:\n%s', '\n'.join(
                    ['%s: %.6f' % (name, score) for score, name in ranked_closest]))

        # get highest ranked theorems from pretrained model
        ret = []
        thm_scores = None

        for tactic_id in top_tacs:
            tactic = self.tactics[tactic_id]

            if (thm_scores is None or self.model_architecture == 'PARAMETERS_CONDITIONED_ON_TAC'):
                thm_scores = self._get_theorem_scores(proof_state_enc, thm_number,
                                                      tactic_id)

                no_params_score = self.predictor.batch_thm_scores(
                    proof_state_enc, empty_emb_batch, tactic_id)[0]

                logger.debug('Theorem score for empty theorem: %f0.2',
                             no_params_score)

            thm_ranked = sorted(
                zip(thm_scores, self.thm_names),
                reverse=True)[:self.options.max_theorem_parameters]

            pass_no_arguments = thm_ranked[-1][0] < no_params_score

            # mix in additional theorems based on similarity wrt self.compute_closest
            thm_ranked = self.add_similar(thm_ranked, ranked_closest)

            logger.debug('thm_ranked: %s', str(thm_ranked))
            tactic_str = str(tactic.name)
            try:
                tactic_params = _compute_parameter_string(
                    list(tactic.parameter_types), pass_no_arguments, thm_ranked)

                for params_str in tactic_params:
                    # todo log softmax for scores?
                    ret.append((tactic_str + params_str, tactic_scores[tactic_id]))

            except ValueError as e:
                logger.debug('Failed to compute parameters for tactic %s: %s',
                             tactic.name, str(e))

        return ret


class MesonActionGenerator(object):
    """Trivial action generator, which always returns MESON tactic."""

    def step(self, goal: proof_assistant_pb2.Theorem,
             thm: proof_assistant_pb2.Theorem) -> List[Tuple[Text, float]]:
        del goal  # unused
        del thm  # unused
        return [('ASM_MESON_TAC [ ]', 1.0)]
