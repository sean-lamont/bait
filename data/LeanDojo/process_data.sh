cd "$(dirname "${BASH_SOURCE[0]}")"
python scripts/download_data.py && python scripts/trace_repos.py