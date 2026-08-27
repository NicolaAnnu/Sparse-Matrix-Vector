#!/bin/bash
set -euo pipefail

source ./common_config_mpi.sh

N=$DEFAULT_N
NZ=$DEFAULT_NZ
MODE=$DEFAULT_MODE
SEED=$DEFAULT_SEED
BLOCK_SIZE=$DEFAULT_MPI_BLOCK_SIZE

NODES=8

MPI_PROCESSES=(128 64 32 16 8)
THREAD_COUNTS=(1 2 4 8 16)

RAW_OUT="$RESULTS_DIR/MPI_hybrid_balance_sweep_raw.csv"
OUT="$RESULTS_DIR/MPI_hybrid_balance_sweep.csv"

echo "mode,n,nz,seed,nodes,mpi_processes,threads_per_process,block_size,run,time,distribution,spmv,normalize,rotate,communication,reduction,epoch" > "$RAW_OUT"

echo "mode,n,nz,seed,nodes,mpi_processes,threads_per_process,block_size,time_med,time_min,time_max,distribution_med,spmv_med,normalize_med,rotate_med,communication_med,reduction_med,epoch_med" > "$OUT"

for i in "${!MPI_PROCESSES[@]}"; do

    mpi_processes="${MPI_PROCESSES[$i]}"
    threads="${THREAD_COUNTS[$i]}"

    echo
    echo "Hybrid balance sweep: nodes=$NODES ranks=$mpi_processes threads/rank=$threads"

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

        output=$(run_mpi_rank_thread "$NODES" "$mpi_processes" "$threads" \
            -n "$N" \
            -nz "$NZ" \
            -m "$MODE" \
            -s "$SEED" \
            -b "$BLOCK_SIZE")

        echo "$output"

        time=$(echo "$output" | extract_time)
        distribution=$(echo "$output" | extract_distribution_time)
        spmv=$(echo "$output" | extract_spmv_time)
        normalize=$(echo "$output" | extract_normalize_time)
        rotate=$(echo "$output" | extract_rotate_time)
        communication=$(echo "$output" | extract_communication_time)
        reduction=$(echo "$output" | extract_reduction_time)
        epoch=$(echo "$output" | extract_epoch_time)

        times+=("$time")
        distributions+=("$distribution")
        spmvs+=("$spmv")
        normalizations+=("$normalize")
        rotations+=("$rotate")
        communications+=("$communication")
        reductions+=("$reduction")
        epochs+=("$epoch")

        echo "$MODE,$N,$NZ,$SEED,$NODES,$mpi_processes,$threads,$BLOCK_SIZE,$r,$time,$distribution,$spmv,$normalize,$rotate,$communication,$reduction,$epoch" >> "$RAW_OUT"

    done

    time_med=$(calculate_median "${times[@]}")
    time_min=$(printf "%s\n" "${times[@]}" | sort -n | head -1)
    time_max=$(printf "%s\n" "${times[@]}" | sort -n | tail -1)

    distribution_med=$(calculate_median "${distributions[@]}")
    spmv_med=$(calculate_median "${spmvs[@]}")
    normalize_med=$(calculate_median "${normalizations[@]}")
    rotate_med=$(calculate_median "${rotations[@]}")
    communication_med=$(calculate_median "${communications[@]}")
    reduction_med=$(calculate_median "${reductions[@]}")
    epoch_med=$(calculate_median "${epochs[@]}")

    echo "$MODE,$N,$NZ,$SEED,$NODES,$mpi_processes,$threads,$BLOCK_SIZE,$time_med,$time_min,$time_max,$distribution_med,$spmv_med,$normalize_med,$rotate_med,$communication_med,$reduction_med,$epoch_med" >> "$OUT"

    echo "Median: $time_med s"
    echo "Min:    $time_min s"
    echo "Max:    $time_max s"

done

echo
echo "Raw results: $RAW_OUT"
echo "Summary:     $OUT"