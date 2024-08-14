#!/bin/bash

TIMEOUT=90

# fix 100 seeds for the RNG for reproducibility of results
SEEDS="7347 7945 1788 5178 3923 130 1077 1815 7455 801
4916 5959 3741 596 9770 8351 9936 1482 7252 3152
2201 551 4748 6911 4221 6421 485 9791 572 7642
2592 9420 5852 9092 6528 4826 3497 3132 4321 2274
3988 6254 271 8196 9335 1582 9784 7887 4842 1308
1162 700 351 1151 5929 8722 1028 3208 8935 7183
5404 3028 954 8436 2637 4684 8459 510 1714 7113
2733 4596 1299 5057 4005 6811 9948 9318 454 9235
4436 4974 5190 6206 3062 9879 2261 9434 199 7870
5596 9656 6169 9108 4921 3260 9426 7866 7712 2243"

MIN_QUBIT=10
MAX_QUBIT=30
QUBIT_STEP=1

mkdir -p data_iqp
cd data_iqp

timeout_count=0
for qubit in $(seq $MIN_QUBIT $QUBIT_STEP $MAX_QUBIT); do
    if [ $timeout_count -ge 10 ]; then break; fi
    timeout_count=0
    for seed in $SEEDS; do
        file="iqp_${qubit}_${seed}"
        echo $file
        if [ -f $file ]; then
            echo "EXISTS $file"
        else
            timeout $TIMEOUT ../../target/release/examples/iqp_stabrank $qubit $seed
            if [ $? == 124 ]; then 
                echo "TIMEOUT $file\n"
                timeout_count=$((timeout_count + 1))
                if [ $timeout_count -ge 10 ]; then
                    echo "10 TIMEOUTS IN A ROW, SKIPPING QUBIT $qubit\n"
                    break
                fi
            else
                timeout_count=0
            fi
        fi
    done
done
