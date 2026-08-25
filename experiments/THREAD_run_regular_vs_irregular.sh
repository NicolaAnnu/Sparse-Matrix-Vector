#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_BLOCK_SIZE

MODES=("regular" "irregular")
THREAD_COUNTS=(16 32)

BASELINE_FILE="$RESULTS_DIR/THREAD_baseline_speedup.csv"
OUT="$RESULTS_DIR/THREAD_regular_vs_irregular.csv"

echo "n,nz,mode,seed,threads,block_size,seq_time_med,thread_time_med,speedup,efficiency_percent" > "$OUT"

for mode in "${MODES[@]}"; do

    if [[ "$mode" == "regular" ]]; then

        echo "Calculating sequential baseline for regular matrix"

        seq_times=()

        for r in $(seq 1 "$REPEATS"); do
            echo "Sequential regular run $r/$REPEATS"

            output=$(run_seq \
                -n "$N" \
                -nz "$NZ" \
                -m "$mode" \
                -s "$SEED")

            echo "$output"

            seq_times+=("$(echo "$output" | extract_time)")
        done

        seq_med=$(calculate_median "${seq_times[@]}")

        echo "Sequential regular median: $seq_med s"

    else

        seq_med=$(awk -F, \
            -v n="$N" \
            -v nz="$NZ" \
            -v mode="$mode" \
            -v seed="$SEED" \
            'NR > 1 && $1 == n && $2 == nz && $3 == mode && $4 == seed {
                print $7
                exit
            }' "$BASELINE_FILE")
            
        echo "Sequential irregular median recuperata: $seq_med s"
    fi

    # Test C++ threads
    for threads in "${THREAD_COUNTS[@]}"; do

        echo "Testing mode=$mode threads=$threads"

        times=()

        for r in $(seq 1 "$REPEATS"); do
            echo "Run $r/$REPEATS"

            output=$(run_cpp_threads "$threads" \
                -n "$N" \
                -nz "$NZ" \
                -m "$mode" \
                -s "$SEED" \
                -b "$BLOCK_SIZE")

            echo "$output"

            times+=("$(echo "$output" | extract_time)")
        done

        thread_med=$(calculate_median "${times[@]}")

        speedup=$(awk \
            -v s="$seq_med" \
            -v p="$thread_med" \
            'BEGIN {printf "%.6f", s/p}')

        efficiency=$(awk \
            -v s="$speedup" \
            -v t="$threads" \
            'BEGIN {printf "%.4f", 100*s/t}')

        echo "$N,$NZ,$mode,$SEED,$threads,$BLOCK_SIZE,$seq_med,$thread_med,$speedup,$efficiency" >> "$OUT"
    done
done

echo "Results: $OUT"