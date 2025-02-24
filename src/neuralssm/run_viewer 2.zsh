#!/bin/zsh

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start_trial> <end_trial> <overwrite> <experiment file>"
    exit 1
fi

START_TRIAL=$1
END_TRIAL=$2
OVERWRT=$3
EXPERIMENT_FILE=$4

# Convert OVERWRT to a proper boolean value (1/0) for Python
if [[ "$OVERWRT" == "true" ]]; then
    BOOL_OVERWRT="true"
else
    BOOL_OVERWRT="false"
fi

# Loop through the specified trial range
for trial in {$START_TRIAL..$END_TRIAL}
do
    echo "Running trial $trial with experiment file $EXPERIMENT_FILE..."
    python3 main.py view -t "$trial" -o "$BOOL_OVERWRT" "$EXPERIMENT_FILE"
done

echo "All trials completed."