"""Tests for deepmath.prover.hol_light.theorem_fingerprint."""

# import tensorflow as tf
from data.HOList.utils import theorem_fingerprint
from environments.HOList.proof_assistant import proof_assistant_pb2

import unittest


class TheoremFingerprintTest(unittest.TestCase):

  def testStableFingerprint(self):
    """Tests that the theorem fingerprint function is stable."""
    theorem = proof_assistant_pb2.Theorem(
        conclusion="concl", hypotheses=["hyp1", "hyp2"])
    self.assertEqual(
        theorem_fingerprint.Fingerprint(theorem), 198703484454304307)
    self.assertEqual(
        theorem_fingerprint.ToTacticArgument(theorem), "THM 198703484454304307")


if __name__ == "__main__":
    unittest.main()
