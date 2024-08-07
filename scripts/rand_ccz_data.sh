#!/bin/bash

TIMEOUT=60

# fix 50 seeds for the RNG for reproducibility of results
SEEDS="7347 7945 1788 5178 3923 130 1077 1815 7455 801
4916 5959 3741 596 9770 8351 9936 1482 7252 3152
2201 551 4748 6911 4221 6421 485 9791 572 7642
2592 9420 5852 9092 6528 4826 3497 3132 4321 2274
3988 6254 271 8196 9335 1582 9784 7887 4842 1308"

DEPTHS="10 13 16 20 22 25 28 30 34 37 40 43 46 50 52 55 58 60 64 67 70 73 76 80 82 85 88 90 94 97 100 103"
# DEPTHS="10 20 30 40 50 60 70 80 90 100"
QUBIT_COUNTS="8 19 20 50 100"

mkdir -p data
cd data


for qubit in $QUBIT_COUNTS; do
    for depth in $DEPTHS; do
        if ([ "$depth" -gt 45 ] && [ "$qubit" -lt 21 ]) ||
           ([ "$depth" -gt 80 ] && [ "$qubit" -lt 101 ]); then
            continue
        fi
        # if ([ "$depth" -gt 55 ] && [ "$qubit" -lt 21 ]) ||
        #    ([ "$depth" -gt 90 ] && [ "$qubit" -lt 101 ]); then
        #     continue
        # fi
        for seed in $SEEDS; do
            file="rand_ccz_${qubit}_${depth}_${seed}"
            if [ "$qubit" -gt 50 ]; then
                actual_depth=$(( (depth + 200) / 2 ))
                file="rand_ccz_${qubit}_${actual_depth}_${seed}"
            fi
            echo $file
            if [ -f $file ]; then
                echo "EXISTS $file"
            else
                timeout $TIMEOUT ../../target/release/examples/rand_ccz_stabrank $qubit $depth $seed
                if [ $? == 124 ]; then echo "TIMEOUT $file\n"; fi
            fi
        done
    done
done
