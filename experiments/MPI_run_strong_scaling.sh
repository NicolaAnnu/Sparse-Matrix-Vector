#!/bin/bash
set -euo pipefail

source ./common_config_mpi.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREADS=$DEFAULT_MPI_THREADS
BLOCK_SIZE=$DEFAULT_MPI_BLOCK_SIZE

NODE_COUNTS=(1 2 4 6 8)

BASELINE_FILE="$RESULTS_DIR/THREAD_baseline_speedup.csv"

OUT="$RESULTS_DIR/MPI_strong_scaling.csv"

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Error: missing $BASELINE_FILE"
    exit 1
fi

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
    echo "Error: sequential baseline not found"
    exit 1
fi

echo "Sequential baseline: $seq_med s"

echo "n,nz,mode,seed,nodes,mpi_processes,threads_per_process,block_size,seq_time_med,mpi_time_med,absolute_speedup,relative_speedup,relative_efficiency_percent,distribution_med,local_computation_med,communication_med,reduction_med,epoch_med" > "$OUT"

mpi_one_node=""

for nodes in "${NODE_COUNTS[@]}"; do

    echo
    echo "Strong scaling: $nodes node(s)"

    times=()
    distributions=()
    computations=()
    communications=()
    reductions=()
    epochs=()

    for r in $(seq 1 "$REPEATS"); do

        echo "Run $r/$REPEATS"

        output=$(run_mpi "$nodes" "$THREADS" \
            -n "$N" \
            -nz "$NZ" \
            -m "$MODE" \
            -s "$SEED" \
            -b "$BLOCK_SIZE")

        echo "$output"

        times+=("$(echo "$output" | extract_time)")
        distributions+=("$(echo "$output" | extract_distribution_time)")
        computations+=("$(echo "$output" | extract_local_computation_time)")
        communications+=("$(echo "$output" | extract_communication_time)")
        reductions+=("$(echo "$output" | extract_reduction_time)")
        epochs+=("$(echo "$output" | extract_epoch_time)")
    done

    time_med=$(calculate_median "${times[@]}")
    distribution_med=$(calculate_median "${distributions[@]}")
    computation_med=$(calculate_median "${computations[@]}")
    communication_med=$(calculate_median "${communications[@]}")
    reduction_med=$(calculate_median "${reductions[@]}")
    epoch_med=$(calculate_median "${epochs[@]}")

    if [[ -z "$mpi_one_node" ]]; then
        mpi_one_node="$time_med"
    fi

    absolute_speedup=$(awk \
        -v seq="$seq_med" \
        -v mpi="$time_med" \
        'BEGIN {printf "%.6f", seq/mpi}')

    relative_speedup=$(awk \
        -v one="$mpi_one_node" \
        -v mpi="$time_med" \
        'BEGIN {printf "%.6f", one/mpi}')

    relative_efficiency=$(awk \
        -v s="$relative_speedup" \
        -v n="$nodes" \
        'BEGIN {printf "%.4f", 100*s/n}')

    echo "$N,$NZ,$MODE,$SEED,$nodes,$nodes,$THREADS,$BLOCK_SIZE,$seq_med,$time_med,$absolute_speedup,$relative_speedup,$relative_efficiency,$distribution_med,$computation_med,$communication_med,$reduction_med,$epoch_med" >> "$OUT"

    echo "MPI median: $time_med s"
    echo "Absolute speedup: $absolute_speedup"
    echo "Relative speedup: $relative_speedup"
    echo "Relative efficiency: $relative_efficiency %"

done

echo
echo "Results: $OUT"