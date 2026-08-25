#!/bin/bash
set -euo pipefail
source ./common_config_openmp.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREAD_COUNTS=(16 32)

BLOCK_SIZES=(256 512 1024 2048 4096 8192)

THREAD_FILE="$RESULTS_DIR/THREAD_task_granularity.csv"

OUT="$RESULTS_DIR/OPENMP_task_granularity.csv"

echo "n,nz,mode,seed,threads,block_size,num_tasks,thread_time_med,openmp_time_med,openmp_vs_threads" > "$OUT"


for threads in "${THREAD_COUNTS[@]}"; do

    for block in "${BLOCK_SIZES[@]}"; do

        echo
        echo "Testing OpenMP threads=$threads block_size=$block"


        # Recover existing C++ Threads result
        # THREAD_task_granularity.csv:
        # n,nz,mode,seed,threads,block_size,num_chunks,time_med
        thread_time=$(awk -F, \
            -v n="$N" \
            -v nz="$NZ" \
            -v mode="$MODE" \
            -v seed="$SEED" \
            -v threads="$threads" \
            -v block="$block" \
            'NR > 1 &&
             $1 == n &&
             $2 == nz &&
             $3 == mode &&
             $4 == seed &&
             $5 == threads &&
             $6 == block {
                print $8
                exit
            }' "$THREAD_FILE")

        echo "C++ Threads median recuperata: $thread_time s"

        # Run ONLY OpenMP
        openmp_times=()

        for r in $(seq 1 "$REPEATS"); do

            echo "OpenMP run $r/$REPEATS"

            output=$(run_openmp "$threads" \
                -n "$N" \
                -nz "$NZ" \
                -m "$MODE" \
                -s "$SEED" \
                -b "$block")

            echo "$output"

            openmp_times+=("$(echo "$output" | extract_time)")

        done

        # OpenMP median

        openmp_med=$(calculate_median "${openmp_times[@]}")


        # Number of OpenMP SpMV tasks
        # One chunk corresponds to one OpenMP task.

        num_tasks=$((N / block + (N % block != 0 ? 1 : 0)))


        # Direct comparison:
        # > 1 -> OpenMP faster
        # < 1 -> C++ Threads faster

        openmp_vs_threads=$(awk \
            -v thread="$thread_time" \
            -v omp="$openmp_med" \
            'BEGIN {
                printf "%.6f", thread/omp
            }')

        echo "$N,$NZ,$MODE,$SEED,$threads,$block,$num_tasks,$thread_time,$openmp_med,$openmp_vs_threads" >> "$OUT"

        echo "Number of tasks: $num_tasks"
        echo "OpenMP median: $openmp_med s"
        echo "OpenMP vs Threads: $openmp_vs_threads"

    done

done


echo
echo "Results: $OUT"