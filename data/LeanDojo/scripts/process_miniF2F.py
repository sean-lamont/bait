from lean_dojo import *

import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union

SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str


import lean_dojo

def export_proofs(splits: Dict[SPLIT_STRATEGY, SPLIT], dst_path: Path) -> None:
    """Export all proofs in a traced repo to ``dst_path''."""
    for strategy, split in splits.items():
        split_dir = dst_path / strategy
        split_dir.mkdir(parents=True)

        for name, theorems in split.items():
            data = []
            num_tactics = 0

            for thm in theorems:
                tactics = [
                    {
                        "tactic": t.tactic,
                        "annotated_tactic": t.get_annotated_tactic(),
                        "state_before": t.state_before,
                        "state_after": t.state_after,
                    }
                    for t in thm.get_traced_tactics()
                    if t.state_before != "no goals"
                    and "·" not in t.tactic  # Ignore "·".
                ]
                num_tactics += len(tactics)
                data.append(
                    {
                        "url": thm.repo.url,
                        "commit": thm.repo.commit,
                        "file_path": str(thm.theorem.file_path),
                        "full_name": thm.theorem.full_name,
                        "start": list(thm.start),
                        "end": list(thm.end),
                        "traced_tactics": tactics,
                    }
                )
            oup_path = split_dir / f"{name}.json"
            json.dump(data, oup_path.open("wt"))
            logger.info(
                f"{len(theorems)} theorems and {num_tactics} tactics saved to {oup_path}"
            )


def export_premises(traced_repo: TracedRepo, dst_path: Path) -> None:
    """Export all premise definitions in a traced repo to ``dst_path``."""
    oup_path = dst_path / "corpus.jsonl"
    num_premises = 0

    with oup_path.open("wt") as oup:
        G = traced_repo.traced_files_graph

        for tf_node in reversed(list(nx.topological_sort(G))):
            tf = G.nodes[tf_node]["traced_file"]
            imports = [str(_) for _ in G.successors(tf_node)]
            premises = tf.get_premise_definitions()
            num_premises += len(premises)
            oup.write(
                json.dumps(
                    {"path": str(tf.path), "imports": imports, "premises": premises}
                )
                + "\n"
            )
    logger.info(
        f"{num_premises} theorems/definitions from {len(traced_repo.traced_files)} files saved to {oup_path}"
    )


def export_licenses(traced_repo: TracedRepo, dst_path: Path) -> None:
    """Export the licenses of a traced repo and all its dependencies to ``dst_path``."""
    license_dir = dst_path / "licenses"
    license_dir.mkdir()
    all_repos = [traced_repo.repo] + list(traced_repo.dependencies.values())

    for repo in all_repos:
        lic = repo.get_license()
        if lic is None:
            continue
        with (license_dir / repo.name).open("wt") as oup:
            oup.write(lic)

    with (license_dir / "README.md").open("wt") as oup:
        oup.write(
            "This directory contains licenses of Lean repos used to generate this dataset. The dataset itself is released under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)."
        )


def export_metadata(traced_repo: TracedRepo, dst_path: Path, **kwargs) -> None:
    """Export the metadata of a traced repo to ``dst_path''."""
    metadata = dict(kwargs)
    metadata["creation_time"] = str(datetime.now())
    metadata["from_repo"] = {
        "url": traced_repo.repo.url,
        "commit": traced_repo.repo.commit,
    }
    metadata["leandojo_version"] = lean_dojo.__version__
    json.dump(metadata, (dst_path / "metadata.json").open("wt"))



def export_data(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    **kwargs,
) -> None:
    """Export a traced repo whose theorems have been splitted to ``dst_path``."""
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    if dst_path.exists():
        logger.warning(f"{dst_path} already exists. Removing it now.")
        shutil.rmtree(dst_path)

    # Export the proofs.
    export_proofs(splits, dst_path)

    # Export the premises (theorems, definitions, etc.).
    export_premises(traced_repo, dst_path)

    # Export the licenses.
    export_licenses(traced_repo, dst_path)

    # Export metadata.
    export_metadata(traced_repo, dst_path, **kwargs)

minif2f = LeanGitRepo(
    "https://github.com/facebookresearch/miniF2F",
    "5271ddec788677c815cf818a06f368ef6498a106",
)
traced_minif2f = trace(minif2f)

splits = {"default": {"val": [], "test": []}}

for tf in traced_minif2f.get_traced_theorems():
    if tf.repo.name != "miniF2F":
        continue
    if tf.file_path.name == "valid.lean":
        splits["default"]["val"].append(tf)
    else:
        assert tf.file_path.name == "test.lean"
        splits["default"]["test"].append(tf)

export_data(
    traced_minif2f, splits, "../data/leandojo_minif2f", dataset_name="LeanDojo MiniF2F"
)