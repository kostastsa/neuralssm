#!/bin/zsh

# Check if the correct number of arguments is provided

START_TRIAL="$1"
END_TRIAL="$2"
OVERWRT="$3"

# EXPERIMENT_FILE=$4
for file in "${@[4,-1]}"; do
    echo "processing $file"

    # if file == $1 || file == $2 || file == $3; then
    #     continue
    # fi

    # Convert OVERWRT to a proper boolean value (1/0) for Python
    if [[ "$OVERWRT" == "true" ]]; then
        BOOL_OVERWRT="true"
    else
        BOOL_OVERWRT="false"
    fi

    # Loop through the specified trial range
    for trial in {$START_TRIAL..$END_TRIAL}
        do
            echo "Running trial $trial with experiment file $file..."
            python3 main.py view -t "$trial" -o "$BOOL_OVERWRT" "$file"
        done

done

echo "All trials completed."