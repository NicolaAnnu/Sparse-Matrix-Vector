#!/bin/bash
set -euo pipefail
source ./common_config_openmp.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_BLOCK_SIZE

THREAD_COUNTS=(1 2 4 8 16 32)

BASELINE_FILE="$RESULTS_DIR/THREAD_baseline_speedup.csv"
THREAD_FILE="$RESULTS_DIR/THREAD_scaling.csv"

OUT="$RESULTS_DIR/OPENMP_scaling.csv"

seq_med=$(awk -F, \
    -v n="$N" \
    -v nz="$NZ" \
    -v mode="$MODE" \
    -v seed="$SEED" \
    'NR > 1 &&
     $1 == n &&
     $2 == nz &&
     $3 == mode &&
     $4 == seed {
        print $7
        exit
    }' "$BASELINE_FILE")

echo "Sequential median recuperata: $seq_med s"

echo "n,nz,mode,seed,block_size,threads,seq_time_med,thread_time_med,openmp_time_med,thread_speedup,openmp_speedup,openmp_efficiency_percent,openmp_vs_threads" > "$OUT"


for threads in "${THREAD_COUNTS[@]}"; do

    echo
    echo "Testing OpenMP with $threads threads"

    thread_time=$(awk -F, \
        -v n="$N" \
        -v nz="$NZ" \
        -v mode="$MODE" \
        -v seed="$SEED" \
        -v block="$BLOCK_SIZE" \
        -v threads="$threads" \
        'NR > 1 &&
         $1 == n &&
         $2 == nz &&
         $3 == mode &&
         $4 == seed &&
         $5 == block &&
         $6 == threads {
            print $8
            exit
        }' "$THREAD_FILE")


    echo "C++ Threads median recuperata: $thread_time s"

    openmp_times=()

    for r in $(seq 1 "$REPEATS"); do

        echo "OpenMP run $r/$REPEATS"

        output=$(run_openmp "$threads" \
            -n "$N" \
            -nz "$NZ" \
            -m "$MODE" \
            -s "$SEED" \
            -b "$BLOCK_SIZE")

        echo "$output"

        openmp_times+=("$(echo "$output" | extract_time)")

    done

    openmp_med=$(calculate_median "${openmp_times[@]}")

    thread_speedup=$(awk \
        -v s="$seq_med" \
        -v p="$thread_time" \
        'BEGIN {
            printf "%.6f", s/p
        }')

    openmp_speedup=$(awk \
        -v s="$seq_med" \
        -v p="$openmp_med" \
        'BEGIN {
            printf "%.6f", s/p
        }')

    openmp_efficiency=$(awk \
        -v s="$openmp_speedup" \
        -v t="$threads" \
        'BEGIN {
            printf "%.4f", 100*s/t
        }')

    openmp_vs_threads=$(awk \
        -v thread="$thread_time" \
        -v omp="$openmp_med" \
        'BEGIN {
            printf "%.6f", thread/omp
        }')

    echo "$N,$NZ,$MODE,$SEED,$BLOCK_SIZE,$threads,$seq_med,$thread_time,$openmp_med,$thread_speedup,$openmp_speedup,$openmp_efficiency,$openmp_vs_threads" >> "$OUT"

    echo "OpenMP median: $openmp_med s"
    echo "OpenMP speedup: $openmp_speedup"
    echo "OpenMP vs Threads: $openmp_vs_threads"

done

echo
echo "Results: $OUT"