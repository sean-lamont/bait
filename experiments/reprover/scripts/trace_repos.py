import json
import argparse
import sys
from glob import glob
from loguru import logger
from lean_dojo import LeanGitRepo, trace, is_available_in_cache

# todo place in data/leandojo/..
# todo combine all setup into single script

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    url_commits = set()
    for path in glob(f"{args.data_path}/*/*/*.json"):
        data = json.load(open(path))
        for ex in data:
            url_commits.add((ex["url"], ex["commit"]))

    repos = set()
    for url, commit in url_commits:
        repo = LeanGitRepo(url, commit)
        if not is_available_in_cache(repo):
            repos.add(repo)

    logger.info(f"Repos to trace: {repos}")

    for repo in repos:
        trace(repo)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    main()
