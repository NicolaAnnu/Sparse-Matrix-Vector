#!/bin/bash
set -euo pipefail

source ./common_config_mpi.sh

BASE_N=125000
BASE_NZ=25000000

MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

THREADS=$DEFAULT_MPI_THREADS
BLOCK_SIZE=$DEFAULT_MPI_BLOCK_SIZE

NODE_COUNTS=(1 2 4 6 8)

OUT="$RESULTS_DIR/MPI_weak_scaling.csv"

echo "nodes,mpi_processes,threads_per_process,n,nz,mode,seed,block_size,time_med,weak_efficiency,distribution_med,local_computation_med,spmv_med,normalize_med,rotate_med,communication_med,reduction_med,epoch_med" > "$OUT"

reference_time=""

for nodes in "${NODE_COUNTS[@]}"; do

    N=$((BASE_N * nodes))
    NZ=$((BASE_NZ * nodes))

    echo
    echo "Weak scaling: nodes=$nodes N=$N NZ=$NZ"

    times=()
    distributions=()
    spmvs=()
    normalizations=()
    rotations=()
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
        spmvs+=("$(echo "$output" | extract_spmv_time)")
        normalizations+=("$(echo "$output" | extract_normalize_time)")
        rotations+=("$(echo "$output" | extract_rotate_time)")
        communications+=("$(echo "$output" | extract_communication_time)")
        reductions+=("$(echo "$output" | extract_reduction_time)")
        epochs+=("$(echo "$output" | extract_epoch_time)")
    done

    time_med=$(calculate_median "${times[@]}")
    distribution_med=$(calculate_median "${distributions[@]}")
    spmv_med=$(calculate_median "${spmvs[@]}")
    normalize_med=$(calculate_median "${normalizations[@]}")
    rotate_med=$(calculate_median "${rotations[@]}")
    communication_med=$(calculate_median "${communications[@]}")
    reduction_med=$(calculate_median "${reductions[@]}")
    epoch_med=$(calculate_median "${epochs[@]}")

    if [[ -z "$reference_time" ]]; then
        reference_time="$time_med"
    fi

    weak_efficiency=$(awk \
        -v ref="$reference_time" \
        -v current="$time_med" \
        'BEGIN {printf "%.6f", ref/current}')

    echo "$nodes,$nodes,$THREADS,$N,$NZ,$MODE,$SEED,$BLOCK_SIZE,$time_med,$weak_efficiency,$distribution_med,$computation_med,$spmv_med,$normalize_med,$rotate_med,$communication_med,$reduction_med,$epoch_med" >> "$OUT"

    echo "MPI median: $time_med s"
    echo "Weak efficiency: $weak_efficiency"

done

echo
echo "Results: $OUT"