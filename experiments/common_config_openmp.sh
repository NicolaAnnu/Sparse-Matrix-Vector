#!/bin/bash

# Executables

OPENMP_BIN="../openmp_spmv"
SEQ_BIN="../seq"

# Default experiment parameters

DEFAULT_N=1000000
DEFAULT_NZ=200000000
DEFAULT_MODE="irregular"
DEFAULT_SEED=111
DEFAULT_THREADS=32
DEFAULT_BLOCK_SIZE=1024
REPEATS=3

PARTITION="normal"
TIME_LIMIT="00:15:00"
PHYSICAL_CORES=16
LOGICAL_CPUS=32
RESULTS_DIR="./results"

mkdir -p "$RESULTS_DIR"

extract_time() {
    awk -F'= ' '/^Time \(sec\)/ {print $2; exit}'
}

calculate_median() {
    printf '%s\n' "$@" | sort -n | awk '
    {
        values[NR] = $1
    }
    END {
        if (NR % 2 == 1)
            print values[(NR + 1) / 2]
        else
            print (values[NR / 2] + values[NR / 2 + 1]) / 2
    }'
}

run_seq() {
    srun --partition="$PARTITION" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=1 \
      --hint=nomultithread \
      --cpu-bind=cores \
      --time="$TIME_LIMIT" \
      "$SEQ_BIN" "$@"
}

run_openmp() {
    local threads="$1"
    shift

    if (( threads <= PHYSICAL_CORES )); then
        srun --partition="$PARTITION" --nodes=1 --ntasks=1 \
          --cpus-per-task="$threads" \
          --hint=nomultithread \
          --time="$TIME_LIMIT" \
          env OMP_NUM_THREADS="$threads" \
              OMP_PLACES=cores \
              OMP_PROC_BIND=close \
          "$OPENMP_BIN" "$@" -t "$threads"
    else
        srun --partition="$PARTITION" --nodes=1 --ntasks=1 \
          --cpus-per-task="$threads" \
          --cpu-bind=threads \
          --time="$TIME_LIMIT" \
          env OMP_NUM_THREADS="$threads" \
              OMP_PLACES=threads \
              OMP_PROC_BIND=close \
          "$OPENMP_BIN" "$@" -t "$threads"
    fi
}