#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_BLOCK_SIZE
MODES=("regular" "irregular")
THREAD_COUNTS=(16 32)

OUT="$RESULTS_DIR/regular_vs_irregular.csv"
echo "n,nz,mode,seed,threads,block_size,time_med" > "$OUT"

for mode in "${MODES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "Testing mode=$mode threads=$threads"

        times=()

        for r in $(seq 1 "$REPEATS"); do
            output=$(run_cpp_threads "$threads" \
                -n "$N" -nz "$NZ" -m "$mode" -s "$SEED" -b "$BLOCK_SIZE")

            echo "$output"
            times+=("$(echo "$output" | extract_time)")
        done

        median=$(calculate_median "${times[@]}")
        echo "$N,$NZ,$mode,$SEED,$threads,$BLOCK_SIZE,$median" >> "$OUT"
    done
done

echo "Results: $OUT"
