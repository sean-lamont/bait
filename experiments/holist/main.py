from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

""""

DeepHOL non-distributed prover.

"""

from experiments.holist import prover_flags
from experiments.holist import prover_runner
from absl import flags


def main():
  prover_runner.program_started()
  prover_runner.run_pipeline(*prover_flags.process_prover_flags())


if __name__ == '__main__':
  logging.basicConfig(level=logging.FATAL)

  FLAGS = flags.FLAGS
  FLAGS(sys.argv)

  main()