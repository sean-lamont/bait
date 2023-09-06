"""Simple runner for the prover.

Iterate over the tasks sequentially.

This runner can run the prover on a set of tasks without the overhead of
starting a distributed job.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import List

import wandb
from tqdm import tqdm

from environments.holist import proof_assistant_pb2
from experiments.holist import deephol_pb2
from experiments.holist.prover import prover
from data.holist.utils import stats, io_util


def program_started():
    pass


def compute_stats(output):
    """Compute aggregate statistics given prooflog file."""
    logging.info('Computing aggregate statistics from %s', output)
    stat_list = [
        stats.proof_log_stats(log)
        for log in io_util.read_protos(output, deephol_pb2.ProofLog)
    ]
    if not stat_list:
        logging.info('Empty stats list.')
        return
    aggregate_stat = stats.aggregate_stats(stat_list)
    logging.info('Aggregated statistics:')
    logging.info(stats.aggregate_stat_to_string(aggregate_stat))
    return aggregate_stat


# add config for prediction model
def run_pipeline(prover_tasks: List[proof_assistant_pb2.ProverTask],
                 prover_options: deephol_pb2.ProverOptions, config):
    """Iterate over all prover tasks and store them in the specified file."""

    if config.output.split('.')[-1] != 'textpbs':
        logging.warning('Output file should end in ".textpbs"')

    if config.exp_config.resume:
        logging.info("Resuming run..")
        proof_logs = [log for log in io_util.read_protos(config.output, deephol_pb2.ProofLog)]
        tasks_done = [log for log in io_util.read_protos(config.done_output, proof_assistant_pb2.ProverTask)]

        prover_tasks = [task for task in prover_tasks if task not in tasks_done]

    else:
        proof_logs = []
        tasks_done = []

    prover.cache_embeddings(prover_options, config)
    this_prover = prover.create_prover(prover_options, config)


    logging.info(f"Evaluating {len(prover_tasks)} goals..")

    def save_and_log():
        logging.info('Writing %d proof logs as text proto to %s',
                     len(proof_logs), config.output)

        io_util.write_text_protos(config.output, proof_logs)

        io_util.write_text_protos(config.done_output, tasks_done)

        aggregated_stats = compute_stats(config.output)

        wandb.log({
            'num_theorems_proved': aggregated_stats.num_theorems_proved,
            'num_theorems_attempted': aggregated_stats.num_theorems_attempted,
            'num_theorems_with_bad_proof': aggregated_stats.num_theorems_with_bad_proof,
            'num_reduced_nodes': aggregated_stats.num_reduced_nodes,
            'num_closed_nodes': aggregated_stats.num_closed_nodes,
            'num_nodes': aggregated_stats.num_nodes,
            'time_spent_seconds': aggregated_stats.time_spent_milliseconds / 1000.0,
            'time_spent_days': aggregated_stats.time_spent_milliseconds / (1000.0 * 24 * 60 * 60),
            'total_prediction_time_seconds': aggregated_stats.total_prediction_time / 1000.0})


    for i, task in tqdm(enumerate(prover_tasks)):

        proof_log = this_prover.prove(task)
        proof_logs.append(proof_log)
        tasks_done.append(task)

        if (i + 1) % config.log_frequency == 0:
            save_and_log()

    save_and_log()

    logging.info('Proving complete!')