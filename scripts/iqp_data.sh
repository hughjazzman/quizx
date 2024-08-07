#!/bin/bash

TIMEOUT=60

# fix 50 seeds for the RNG for reproducibility of results
SEEDS="7347 7945 1788 5178 3923 130 1077 1815 7455 801
4916 5959 3741 596 9770 8351 9936 1482 7252 3152
2201 551 4748 6911 4221 6421 485 9791 572 7642
2592 9420 5852 9092 6528 4826 3497 3132 4321 2274
3988 6254 271 8196 9335 1582 9784 7887 4842 1308"

MIN_QUBIT=10
MAX_QUBIT=25
QUBIT_STEP=5

mkdir -p data_iqp
cd data_iqp


for qubit in $(seq $MIN_QUBIT $QUBIT_STEP $MAX_QUBIT); do
    for seed in $SEEDS; do
        file="iqp_${qubit}_${seed}"
        echo $file
        if [ -f $file ]; then
            echo "EXISTS $file"
        else
            timeout $TIMEOUT ../../target/release/examples/iqp_stabrank $qubit $seed
            if [ $? == 124 ]; then echo "TIMEOUT $file\n"; fi
        fi
    done
done
