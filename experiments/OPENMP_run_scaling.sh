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


# Check existing Threads results

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Errore: manca $BASELINE_FILE"
    exit 1
fi

if [[ ! -f "$THREAD_FILE" ]]; then
    echo "Errore: manca $THREAD_FILE"
    exit 1
fi


# Recover sequential baseline already calculated
# THREAD_baseline_speedup.csv:
#
# n,nz,mode,seed,threads,block_size,
# seq_time_med,thread_time_med,speedup,efficiency_percent

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


if [[ -z "$seq_med" ]]; then
    echo "Errore: baseline sequenziale non trovata"
    exit 1
fi


echo "Sequential median recuperata: $seq_med s"


# Output CSV

echo "n,nz,mode,seed,block_size,threads,seq_time_med,thread_time_med,openmp_time_med,thread_speedup,openmp_speedup,openmp_efficiency_percent,openmp_vs_threads" > "$OUT"


for threads in "${THREAD_COUNTS[@]}"; do

    echo
    echo "Testing OpenMP with $threads threads"


    # Recover existing C++ Threads time
    #
    # THREAD_scaling.csv:
    #
    # n,nz,mode,seed,block_size,threads,
    # seq_time_med,thread_time_med,speedup,efficiency_percent

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


    if [[ -z "$thread_time" ]]; then
        echo "Errore: risultato Threads non trovato per threads=$threads"
        exit 1
    fi


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
            -b "$BLOCK_SIZE")

        echo "$output"

        openmp_times+=("$(echo "$output" | extract_time)")

    done


    # OpenMP median

    openmp_med=$(calculate_median "${openmp_times[@]}")


    # Existing C++ Threads speedup

    thread_speedup=$(awk \
        -v s="$seq_med" \
        -v p="$thread_time" \
        'BEGIN {
            printf "%.6f", s/p
        }')


    # OpenMP speedup

    openmp_speedup=$(awk \
        -v s="$seq_med" \
        -v p="$openmp_med" \
        'BEGIN {
            printf "%.6f", s/p
        }')


    # OpenMP efficiency

    openmp_efficiency=$(awk \
        -v s="$openmp_speedup" \
        -v t="$threads" \
        'BEGIN {
            printf "%.4f", 100*s/t
        }')


    # Direct comparison:
    #
    # > 1 -> OpenMP faster
    # < 1 -> C++ Threads faster

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