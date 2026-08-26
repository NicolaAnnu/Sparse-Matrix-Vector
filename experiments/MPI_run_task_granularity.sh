#!/bin/bash
set -euo pipefail

source ./common_config_mpi.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED

NODES=8

THREAD_COUNTS=(16 32)
BLOCK_SIZES=(256 512 1024 2048 4096)

OUT="$RESULTS_DIR/MPI_task_granularity.csv"

echo "n,nz,mode,seed,nodes,mpi_processes,threads_per_process,block_size,time_med,distribution_med,spmv_med,normalize_med,rotate_med,communication_med,reduction_med,epoch_med" > "$OUT"

for threads in "${THREAD_COUNTS[@]}"; do

    for block_size in "${BLOCK_SIZES[@]}"; do

        echo
        echo "MPI granularity: nodes=$NODES threads=$threads block=$block_size"

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

            output=$(run_mpi "$NODES" "$threads" \
                -n "$N" \
                -nz "$NZ" \
                -m "$MODE" \
                -s "$SEED" \
                -b "$block_size")

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

        echo "$N,$NZ,$MODE,$SEED,$NODES,$NODES,$threads,$block_size,$time_med,$distribution_med,$spmv_med,$normalize_med,$rotate_med,$communication_med,$reduction_med,$epoch_med" >> "$OUT"

        echo "Median = $time_med s"

    done

done

echo
echo "Results: $OUT"