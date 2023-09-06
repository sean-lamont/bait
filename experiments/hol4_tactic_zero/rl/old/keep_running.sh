#!/bin/bash
until python3 experiment_gnn_new_goal.py; do
    echo "Process crashed with exit code $?.  Respawning.." >&2
    echo $(date) >> crash_times.txt
    sleep 1
done



