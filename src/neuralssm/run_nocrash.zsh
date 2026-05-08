#!/bin/bash

SAMPLE_GT="$1"
SEED_SETTING="$2"
START_TRIAL="$3"
END_TRIAL="$4"
FILES="$5"
LP_CUTOFF="${6:-}"

while true; do
    if [ -n "$LP_CUTOFF" ]; then
        python3 main.py trials $SAMPLE_GT $SEED_SETTING $START_TRIAL $END_TRIAL $FILES --lp-cutoff $LP_CUTOFF
    else
        python3 main.py trials $SAMPLE_GT $SEED_SETTING $START_TRIAL $END_TRIAL $FILES
    fi
    status=$?
    if [ $status -eq 0 ]; then
        echo "run_trials completed successfully."
        break
    else
        echo "run_trials crashed with exit code $status. Retrying..."
        sleep 1
    fi
done
