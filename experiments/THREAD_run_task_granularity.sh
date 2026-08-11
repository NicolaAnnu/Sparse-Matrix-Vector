#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREAD_COUNTS=(16 32)
BLOCK_SIZES=(256 512 1024 2048 4096 8192)

OUT="$RESULTS_DIR/THREAD_task_granularity.csv"
echo "n,nz,mode,seed,threads,block_size,num_chunks,time_med" > "$OUT"

for threads in "${THREAD_COUNTS[@]}"; do
    for block in "${BLOCK_SIZES[@]}"; do
        echo "Testing threads=$threads block_size=$block"

        times=()

        for r in $(seq 1 "$REPEATS"); do
            echo "Run $r/$REPEATS"

            output=$(run_cpp_threads "$threads" \
                -n "$N" \
                -nz "$NZ" \
                -m "$MODE" \
                -s "$SEED" \
                -b "$block")

            echo "$output"
            times+=("$(echo "$output" | extract_time)")
        done

        median=$(calculate_median "${times[@]}")
        chunks=$((N / block + (N % block != 0 ? 1 : 0)))

        echo "$N,$NZ,$MODE,$SEED,$threads,$block,$chunks,$median" >> "$OUT"
    done
done

echo "Results: $OUT"