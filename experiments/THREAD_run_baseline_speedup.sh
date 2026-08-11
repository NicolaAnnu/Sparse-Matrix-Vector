#!/bin/bash
set -euo pipefail
source ./common_config.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_BLOCK_SIZE
THREAD_COUNTS=(16 32)

OUT="$RESULTS_DIR/THREAD_baseline_speedup.csv"
echo "n,nz,mode,seed,threads,block_size,seq_time_med,thread_time_med,speedup,efficiency_percent" > "$OUT"

seq_times=()

for r in $(seq 1 "$REPEATS"); do
    echo "Sequential run $r/$REPEATS"
    output=$(run_seq -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED")
    echo "$output"
    seq_times+=("$(echo "$output" | extract_time)")
done

seq_med=$(calculate_median "${seq_times[@]}")

for threads in "${THREAD_COUNTS[@]}"; do
    thread_times=()

    for r in $(seq 1 "$REPEATS"); do
        echo "C++ threads $threads, run $r/$REPEATS"
        output=$(run_cpp_threads "$threads" \
            -n "$N" -nz "$NZ" -m "$MODE" -s "$SEED" -b "$BLOCK_SIZE")
        echo "$output"
        thread_times+=("$(echo "$output" | extract_time)")
    done

    thread_med=$(calculate_median "${thread_times[@]}")
    speedup=$(awk -v s="$seq_med" -v p="$thread_med" 'BEGIN {printf "%.6f", s/p}')
    efficiency=$(awk -v s="$speedup" -v t="$threads" 'BEGIN {printf "%.4f", 100*s/t}')

    echo "$N,$NZ,$MODE,$SEED,$threads,$BLOCK_SIZE,$seq_med,$thread_med,$speedup,$efficiency" >> "$OUT"
done

echo "Results: $OUT"