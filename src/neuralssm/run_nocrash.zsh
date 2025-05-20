#!/bin/bash

SAMPLE_GT="$1"
START_TRIAL="$2"
END_TRIAL="$3"
FILES="$4"

while true; do
    python3 main.py trials $1 'r' $2 $3 $4
    status=$?
    if [ $status -eq 0 ]; then
        echo "run_trials completed successfully."
        break
    else
        echo "run_trials crashed with exit code $status. Retrying..."
        sleep 1  # optional: wait a bit before retrying
    fi
done