#!/bin/bash
set -euo pipefail
source ./common_config_openmp.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREAD_BLOCK_SIZE=1024
OPENMP_BLOCK_SIZE=512

THREAD_COUNTS=(1 2 4 8 16 32)

THREAD_FILE="$RESULTS_DIR/THREAD_scaling.csv"
OUT="$RESULTS_DIR/OPENMP_strong_scaling.csv"

declare -A THREAD_TIMES
declare -A OPENMP_TIMES


# Run OpenMP and recover existing C++ Threads results
for threads in "${THREAD_COUNTS[@]}"; do

    echo
    echo "Testing strong scaling with $threads threads"


    # Recover existing C++ Threads result
    thread_time=$(awk -F, \
        -v n="$N" \
        -v nz="$NZ" \
        -v mode="$MODE" \
        -v seed="$SEED" \
        -v block="$THREAD_BLOCK_SIZE" \
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

    if [[ -z "$thread_time" ]]; then
        echo "Error: C++ Threads result not found"
        echo "threads=$threads block_size=$THREAD_BLOCK_SIZE"
        exit 1
    fi

    echo "C++ Threads median recuperata: $thread_time s"

    THREAD_TIMES[$threads]="$thread_time"


    # Run ONLY OpenMP with block size 512
    openmp_times=()

    for r in $(seq 1 "$REPEATS"); do

        echo "OpenMP run $r/$REPEATS"

        output=$(run_openmp "$threads" \
            -n "$N" \
            -nz "$NZ" \
            -m "$MODE" \
            -s "$SEED" \
            -b "$OPENMP_BLOCK_SIZE")

        echo "$output"

        openmp_times+=("$(echo "$output" | extract_time)")

    done

    openmp_med=$(calculate_median "${openmp_times[@]}")

    OPENMP_TIMES[$threads]="$openmp_med"

    echo "OpenMP median: $openmp_med s"

done


# Strong-scaling references
THREAD_T1="${THREAD_TIMES[1]}"
OPENMP_T1="${OPENMP_TIMES[1]}"


echo
echo "C++ Threads T(1): $THREAD_T1 s"
echo "OpenMP T(1): $OPENMP_T1 s"


# CSV header
echo "n,nz,mode,seed,thread_block_size,openmp_block_size,threads,thread_time_med,openmp_time_med,thread_speedup,openmp_speedup,thread_efficiency_percent,openmp_efficiency_percent,openmp_vs_threads" > "$OUT"


for threads in "${THREAD_COUNTS[@]}"; do

    thread_time="${THREAD_TIMES[$threads]}"
    openmp_time="${OPENMP_TIMES[$threads]}"


    # Strong-scaling speedup: T(1) / T(p)
    thread_speedup=$(awk \
        -v t1="$THREAD_T1" \
        -v tp="$thread_time" \
        'BEGIN {
            printf "%.6f", t1/tp
        }')

    openmp_speedup=$(awk \
        -v t1="$OPENMP_T1" \
        -v tp="$openmp_time" \
        'BEGIN {
            printf "%.6f", t1/tp
        }')


    # Parallel efficiency: speedup / p
    thread_efficiency=$(awk \
        -v s="$thread_speedup" \
        -v t="$threads" \
        'BEGIN {
            printf "%.4f", 100*s/t
        }')

    openmp_efficiency=$(awk \
        -v s="$openmp_speedup" \
        -v t="$threads" \
        'BEGIN {
            printf "%.4f", 100*s/t
        }')


    # Direct comparison:
    # > 1 -> OpenMP faster
    # < 1 -> C++ Threads faster
    openmp_vs_threads=$(awk \
        -v thread="$thread_time" \
        -v omp="$openmp_time" \
        'BEGIN {
            printf "%.6f", thread/omp
        }')


    echo "$N,$NZ,$MODE,$SEED,$THREAD_BLOCK_SIZE,$OPENMP_BLOCK_SIZE,$threads,$thread_time,$openmp_time,$thread_speedup,$openmp_speedup,$thread_efficiency,$openmp_efficiency,$openmp_vs_threads" >> "$OUT"


    echo
    echo "threads=$threads"
    echo "C++ Threads median: $thread_time s"
    echo "OpenMP median: $openmp_time s"
    echo "C++ Threads speedup: $thread_speedup"
    echo "OpenMP speedup: $openmp_speedup"
    echo "C++ Threads efficiency: $thread_efficiency %"
    echo "OpenMP efficiency: $openmp_efficiency %"
    echo "OpenMP vs Threads: $openmp_vs_threads"

done


echo
echo "Results: $OUT"