"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""
import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache

from experiments.reprover.common import set_logger
from experiments.reprover.prover_context.proof_search_updown import Status, DistributedProver


def _get_theorems(args) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        args.data_path,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
        data_path: str,
        split: str,
        file_path: Optional[str],
        full_name: Optional[str],
        name_filter: Optional[str],
        num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    cur_url = None
    cur_commit = None
    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
                t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        logger.debug(f'repo {t["url"], t["commit"]}')

        if t['url'] != cur_url or t['commit'] != cur_commit:
            cur_url = t['url']
            cur_commit = t['commit']
            repo = LeanGitRepo(t["url"], t["commit"])

        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))
    theorems = sorted(
        theorems,
        key=lambda t: hashlib.md5(
            (str(t.file_path) + ":" + t.full_name).encode()
        ).hexdigest(),
    )
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint of the tactic generator.",
    )

    parser.add_argument(
        "--goal_path",
        type=str,
        required=True,
        help="Checkpoint of the goal model",
    )

    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        # default=60,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    # Randomly generate an experiment ID if not provided.
    if args.exp_id is None:
        args.exp_id = str(uuid.uuid4())

    set_logger(args.verbose)
    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    repo, theorems, positions = _get_theorems(args)

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        args.ckpt_path,
        args.goal_path,
        args.indexed_corpus_path,
        num_cpus=args.num_cpus,
        with_gpus=args.with_gpus,
        timeout=args.timeout,
        num_sampled_tactics=args.num_sampled_tactics,
        debug=args.verbose,
    )

    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )
    logger.info(f"Pass@1: {num_proved / (num_proved + num_failed)}")

    # Save the results.
    if args.exp_id is not None:
        pickle_path = f"{args.exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")


if __name__ == "__main__":
    main()
