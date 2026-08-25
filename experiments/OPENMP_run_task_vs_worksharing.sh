#!/bin/bash
set -euo pipefail
source ./common_config_openmp.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREAD_COUNTS=(16 32)
BLOCK_SIZES=(256 512 1024 2048 4096 8192)

TASK_FILE="$RESULTS_DIR/OPENMP_task_granularity.csv"

OUT="$RESULTS_DIR/OPENMP_task_vs_worksharing.csv"

echo "n,nz,mode,seed,threads,block_size,num_chunks,task_time_med,worksharing_time_med,worksharing_vs_task" > "$OUT"


for threads in "${THREAD_COUNTS[@]}"; do

    for block in "${BLOCK_SIZES[@]}"; do

        echo
        echo "Testing Task-based vs Work-Sharing threads=$threads block_size=$block"


        # Recover existing OpenMP Task-based result
        # OPENMP_task_granularity.csv:
        # n,nz,mode,seed,threads,block_size,num_tasks,
        # thread_time_med,openmp_time_med,openmp_vs_threads

        task_time=$(awk -F, \
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
                print $9
                exit
            }' "$TASK_FILE")

        if [[ -z "$task_time" ]]; then
            echo "Errore: risultato Task-based non trovato"
            echo "threads=$threads block=$block"
            exit 1
        fi

        echo "Task-based median recuperata: $task_time s"


        # Run ONLY Work-Sharing

        worksharing_times=()

        for r in $(seq 1 "$REPEATS"); do

            echo "Work-Sharing run $r/$REPEATS"

            output=$(run_openmp_worksharing "$threads" \
                -n "$N" \
                -nz "$NZ" \
                -m "$MODE" \
                -s "$SEED" \
                -b "$block")

            echo "$output"

            worksharing_times+=("$(echo "$output" | extract_time)")

        done


        worksharing_med=$(calculate_median "${worksharing_times[@]}")
        # Same logical subdivision of rows.
        # In Task-based these chunks correspond to tasks.
        # In Work-Sharing they are dynamic scheduling chunks.

        num_chunks=$((N / block + (N % block != 0 ? 1 : 0)))
        # Direct comparison:
        # > 1 -> Work-Sharing faster
        # < 1 -> Task-based faster

        worksharing_vs_task=$(awk \
            -v task="$task_time" \
            -v ws="$worksharing_med" \
            'BEGIN {
                printf "%.6f", task/ws
            }')


        echo "$N,$NZ,$MODE,$SEED,$threads,$block,$num_chunks,$task_time,$worksharing_med,$worksharing_vs_task" >> "$OUT"


        echo "Number of chunks: $num_chunks"
        echo "Task-based median: $task_time s"
        echo "Work-Sharing median: $worksharing_med s"
        echo "Work-Sharing vs Task-based: $worksharing_vs_task"

    done

done

echo
echo "Results: $OUT"