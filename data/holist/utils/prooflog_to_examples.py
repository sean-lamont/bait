"""Converter from ProofLog proto to a dictionary with goals, tactics and parameters."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import logging
from typing import Dict, List, Optional, Text, Tuple

from experiments.holist import deephol_pb2
from data.holist.utils import theorem_fingerprint as fp, io_util
from experiments.holist.distributed_loop import options_pb2
from environments.holist import proof_assistant_pb2


class ProofLogToExamples(object):
    """Class for conversion from prooflog protobuf format to examples"""

    def __init__(self, tactic_name_id_map: Dict[Text, int],
                 theorem_database: proof_assistant_pb2.TheoremDatabase,
                 options: options_pb2.ConvertorOptions):

        """Initializer.

        Arguments:
          tactic_name_id_map: mapping from tactic names to ids.
          theorem_database: database containing list of global theorems with splits
          options: options to control forbidden parameters and graph representations
        """
        if options.scrub_parameters not in [
            options_pb2.ConvertorOptions.NOTHING,
            options_pb2.ConvertorOptions.TESTING,
            options_pb2.ConvertorOptions.VALIDATION_AND_TESTING
        ]:
            raise ValueError('Unknown scrub_parameter.')

        self.tactic_name_id_map = tactic_name_id_map
        self.options = options

        self.fingerprint_conclusion_map = {
            fp.Fingerprint(theorem): theorem.conclusion
            for theorem in theorem_database.theorems
        }

        self.forbidden_parameters = set()
        for theorem in theorem_database.theorems:
            if (theorem.tag not in [
                proof_assistant_pb2.Theorem.DEFINITION,
                proof_assistant_pb2.Theorem.TYPE_DEFINITION
            ] and theorem.training_split == proof_assistant_pb2.Theorem.UNKNOWN):
                raise ValueError('need training split information in theorem database.')

            scrub_testsplit_parameters = options.scrub_parameters in [
                options_pb2.ConvertorOptions.TESTING,
                options_pb2.ConvertorOptions.VALIDATION_AND_TESTING
            ]
            scrub_validsplit_parameters = (
                    options.scrub_parameters ==
                    options_pb2.ConvertorOptions.VALIDATION_AND_TESTING)

            if ((theorem.training_split == proof_assistant_pb2.Theorem.TESTING and
                 scrub_testsplit_parameters) or
                    (theorem.training_split == proof_assistant_pb2.Theorem.VALIDATION and
                     scrub_validsplit_parameters)):
                self.forbidden_parameters.add(fp.Fingerprint(theorem))

    def _get_parameter_conclusion(self,
                                  parameter: proof_assistant_pb2.Theorem) -> Text:
        """Get conclusion from fingerprint (prioritized), or one in theorem."""
        if (parameter.HasField('fingerprint') and
                parameter.fingerprint in self.fingerprint_conclusion_map):
            conclusion = self.fingerprint_conclusion_map[parameter.fingerprint]
            if (parameter.HasField('conclusion') and
                    conclusion != parameter.conclusion):
                raise ValueError('conclusion doesn\'t match that in database')
            return conclusion
        if parameter.HasField('conclusion'):
            return parameter.conclusion
        raise ValueError('Neither conclusion present, nor fingerprint %d found'
                         ' in theorem database.' % parameter.fingerprint)

    def _extract_theorem_parameters(
            self, tactic_application: deephol_pb2.TacticApplication
    ) -> Tuple[List[Text], List[Text]]:
        """Extracts parameters of type theorem from a tactic application, if any.

        Note: it might be misleading to call these theorems. If the source is from
        an assumption, the theorem is of the form x |- x. We return x in this case.

        Arguments:
          tactic_application: tactic application to extract the parameters from.

        Returns:
          A pair of (parameters, hard_negatives), where parameters are the
          conclusions of the parameters and hard_negatives are those selected
          parameters that did not contribute to the final outcome. Both are
          preprocessed.
        """
        theorems = []
        hard_negatives = []
        for parameter in tactic_application.parameters:
            if parameter.theorems and not (
                    parameter.parameter_type == deephol_pb2.Tactic.THEOREM or
                    parameter.parameter_type == deephol_pb2.Tactic.THEOREM_LIST):
                raise ValueError('Unexpected theorem parameters or incorrect type.')

            theorems += [
                self._get_parameter_conclusion(theorem)
                for theorem in parameter.theorems
                if fp.Fingerprint(theorem) not in self.forbidden_parameters
            ]
            hard_negatives += [
                self._get_parameter_conclusion(theorem)
                for theorem in parameter.hard_negative_theorems
                if fp.Fingerprint(theorem) not in self.forbidden_parameters
            ]
        return theorems, hard_negatives

    def _proof_step_features(self, goal_proto: proof_assistant_pb2.Theorem,
                             tactic_application: deephol_pb2.TacticApplication
                             ) -> Dict:
        """Compute the basic features of a proof step (goal, tactic, and args)."""
        # preprocessed goal's conclusion's features
        features = {'goal': goal_proto.conclusion}
        tactic_id = self.tactic_name_id_map[tactic_application.tactic]
        theorem_parameters, hard_negatives = self._extract_theorem_parameters(
            tactic_application)
        features.update({'thms': theorem_parameters})

        features.update({
            # preprocessed goal's hypotheses
            'goal_asl': goal_proto.hypotheses,
            # tactic id of tactic application
            'tac_id': tactic_id,
            # Hard (high scoring) negative examples for the parameters that were
            # selected specifically to train against.
            'thms_hard_negatives': hard_negatives,
        })

        return features

    def process_proof_step(self, goal_proto: proof_assistant_pb2.Theorem,
                           tactic_application: deephol_pb2.TacticApplication
                           ):
        """Convert goal,tactic pair to feature dict (for closed goal) or None."""
        if not tactic_application.closed:
            return None
        return self._proof_step_features(goal_proto, tactic_application)

    def process_proof_node(self, proof_node: deephol_pb2.ProofNode):
        for tactic_application in proof_node.proofs:
            example = self.process_proof_step(proof_node.goal, tactic_application)
            if example is not None:
                yield example

    def process_proof_log(self, proof_log: deephol_pb2.ProofLog):
        for proof_node in proof_log.nodes:
            for example in self.process_proof_node(proof_node):
                yield example

    def process_proof_logs(self, proof_logs):
        for proof_log in proof_logs:
            for example in self.process_proof_log(proof_log):
                yield example

    def to_negative_example(self, negative_theorem: proof_assistant_pb2.Theorem
                            ):
        raise NotImplementedError(
            'to_negative_example not implemented for base class.')


def create_processor(
        options: options_pb2.ConvertorOptions,
        theorem_database: Optional[proof_assistant_pb2.TheoremDatabase] = None,
        tactics: Optional[List[deephol_pb2.Tactic]] = None) -> ProofLogToExamples:
    """Factory function for ProofLogToTorch."""

    if theorem_database and options.theorem_database_path:
        raise ValueError(
            'Both thereom database as well as a path to load it from file '
            'provided. Only provide one.')
    if not theorem_database:
        theorem_database = io_util.load_theorem_database_from_file(
            str(options.theorem_database_path))

    if tactics and options.tactics_path:
        raise ValueError('Both tactics as well as a path to load it from '
                         'provided. Only provide one.')
    if not tactics:
        tactics = io_util.load_tactics_from_file(str(options.tactics_path), None)
    tactics_name_id_map = {tactic.name: tactic.id for tactic in tactics}

    if options.replacements_hack:
        logging.warning('Replacements hack is enabled.')
        tactics_name_id_map.update({
            'GEN_TAC': 8,
            'MESON_TAC': 11,
            'CHOOSE_TAC': 34,
        })

    if options.format != options_pb2.ConvertorOptions.HOLPARAM:
        raise ValueError('Unknown options_pb2.ConvertorOptions.TFExampleFormat.')

    return ProofLogToExamples(tactics_name_id_map, theorem_database, options)
