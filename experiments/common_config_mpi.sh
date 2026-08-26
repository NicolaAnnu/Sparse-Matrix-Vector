#!/bin/bash

# Executables

MPI_BIN="../mpi"
SEQ_BIN="../seq"

DEFAULT_N=1000000
DEFAULT_NZ=200000000
DEFAULT_MODE="irregular"
DEFAULT_SEED=111

DEFAULT_MPI_THREADS=16
DEFAULT_MPI_BLOCK_SIZE=2048

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

extract_rayleigh() {
awk -F'=' '/^rayleigh=/ {print $2; exit}'
}

extract_checksum() {
awk -F'=' '/^checksum=/ {print $2; exit}'
}

extract_distribution_time() {
awk -F'=' '/^distribution_time=/ {print $2; exit}'
}

extract_spmv_time() {
awk -F'=' '/^spmv_time=/ {print $2; exit}'
}

extract_normalize_time() {
awk -F'=' '/^normalize_time=/ {print $2; exit}'
}

extract_rotate_time() {
awk -F'=' '/^rotate_time=/ {print $2; exit}'
}

extract_communication_time() {
awk -F'=' '/^communication_time=/ {print $2; exit}'
}

extract_reduction_time() {
awk -F'=' '/^global_reduction_time=/ {print $2; exit}'
}

extract_epoch_time() {
awk -F'=' '/^epoch_transition_time=/ {print $2; exit}'
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
--time="$TIME_LIMIT" \
"$SEQ_BIN" "$@"
}

run_mpi() {
local nodes="$1"
local threads="$2"
shift 2

if (( threads <= PHYSICAL_CORES )); then

    srun --partition="$PARTITION" \
      --nodes="$nodes" \
      --ntasks="$nodes" \
      --ntasks-per-node=1 \
      --cpus-per-task="$threads" \
      --hint=nomultithread \
      --mpi=pmix \
      --time="$TIME_LIMIT" \
      env OMP_NUM_THREADS="$threads" \
          OMP_DYNAMIC=FALSE \
          OMP_PLACES=cores \
          OMP_PROC_BIND=close \
      "$MPI_BIN" "$@" -t "$threads"

else

    srun --partition="$PARTITION" \
      --nodes="$nodes" \
      --ntasks="$nodes" \
      --ntasks-per-node=1 \
      --cpus-per-task="$threads" \
      --cpu-bind=threads \
      --mpi=pmix \
      --time="$TIME_LIMIT" \
      env OMP_NUM_THREADS="$threads" \
          OMP_DYNAMIC=FALSE \
          OMP_PLACES=threads \
          OMP_PROC_BIND=close \
      "$MPI_BIN" "$@" -t "$threads"

fi

}

run_mpi_rank_thread() {
local ranks="$1"
local threads="$2"
shift 2

srun --partition="$PARTITION" \
  --nodes=1 \
  --ntasks="$ranks" \
  --ntasks-per-node="$ranks" \
  --cpus-per-task="$threads" \
  --hint=nomultithread \
  --cpu-bind=cores \
  --mpi=pmix \
  --time="$TIME_LIMIT" \
  env OMP_NUM_THREADS="$threads" \
      OMP_DYNAMIC=FALSE \
      OMP_PLACES=cores \
      OMP_PROC_BIND=close \
  "$MPI_BIN" "$@" -t "$threads"
}