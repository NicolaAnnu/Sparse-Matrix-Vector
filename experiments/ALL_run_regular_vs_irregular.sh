#!/bin/bash
set -euo pipefail

# Load the project's existing launch helpers.
# The function names are distinct (run_cpp_threads, run_openmp, run_mpi),
# while the shared variables/functions have compatible definitions.
source ./common_config.sh
source ./common_config_openmp.sh
source ./common_config_mpi.sh

# Fixed problem size used in the main experiments.
N=1000000
NZ=200000000
SEED=111

MODES=("regular" "irregular")
THREAD_COUNTS=(16 32)
BLOCK_SIZES=(1024 2048 4096)

# MPI+OpenMP: one MPI rank per node, as in common_config_mpi.sh::run_mpi.
MPI_NODES=8
MPI_PROCESSES=8

OUT="$RESULTS_DIR/regular_vs_irregular_all.csv"
FLAT_OUT="$RESULTS_DIR/regular_vs_irregular_all_flat.csv"

# The first output intentionally follows the sectioned format of the reference
# file shown in the comparison example.
: > "$OUT"

# The second output is a standard CSV, easier to load with pandas/Excel.
echo "implementation,mode,n,nz,nodes,mpi_processes,threads,block_size,total_time_med" > "$FLAT_OUT"

run_and_collect() {
    local implementation="$1"
    local mode="$2"
    local threads="$3"
    local block_size="$4"

    local times=()
    local output=""
    local t=""

    for r in $(seq 1 "$REPEATS"); do
        echo "Run $r/$REPEATS" >&2

        case "$implementation" in
            cpp_threads)
                output=$(run_cpp_threads "$threads" \
                    -n "$N" \
                    -nz "$NZ" \
                    -m "$mode" \
                    -s "$SEED" \
                    -b "$block_size")
                ;;

            openmp)
                output=$(run_openmp "$threads" \
                    -n "$N" \
                    -nz "$NZ" \
                    -m "$mode" \
                    -s "$SEED" \
                    -b "$block_size")
                ;;

            mpi_openmp)
                output=$(run_mpi "$MPI_NODES" "$threads" \
                    -n "$N" \
                    -nz "$NZ" \
                    -m "$mode" \
                    -s "$SEED" \
                    -b "$block_size")
                ;;

            *)
                echo "Error: unknown implementation '$implementation'" >&2
                exit 1
                ;;
        esac

        echo "$output" >&2
        t=$(echo "$output" | extract_time)

        if [[ -z "$t" ]]; then
            echo "Error: could not extract execution time for implementation=$implementation mode=$mode threads=$threads block=$block_size" >&2
            exit 1
        fi

        times+=("$t")
    done

    calculate_median "${times[@]}"
}

# -----------------------------------------------------------------------------
# C++ THREADS
# -----------------------------------------------------------------------------
echo "cpp threads" >> "$OUT"
echo "mode,n,nz,threads,block_size,Total_Time_Med" >> "$OUT"

for mode in "${MODES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            echo
            echo ">> C++ Threads: mode=$mode threads=$threads block=$block_size (repeats=$REPEATS)"

            med_tot=$(run_and_collect cpp_threads "$mode" "$threads" "$block_size")

            echo "$mode,$N,$NZ,$threads,$block_size,$med_tot" >> "$OUT"
            echo "cpp_threads,$mode,$N,$NZ,1,1,$threads,$block_size,$med_tot" >> "$FLAT_OUT"

            echo "  -> Median Total Time: ${med_tot}s"
        done
    done
done

# -----------------------------------------------------------------------------
# OPENMP TASK-BASED
# -----------------------------------------------------------------------------
echo >> "$OUT"
echo "OMP" >> "$OUT"
echo "mode,n,nz,threads,block_size,Total_Time_Med" >> "$OUT"

for mode in "${MODES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            echo
            echo ">> OpenMP: mode=$mode threads=$threads block=$block_size (repeats=$REPEATS)"

            med_tot=$(run_and_collect openmp "$mode" "$threads" "$block_size")

            echo "$mode,$N,$NZ,$threads,$block_size,$med_tot" >> "$OUT"
            echo "openmp,$mode,$N,$NZ,1,1,$threads,$block_size,$med_tot" >> "$FLAT_OUT"

            echo "  -> Median Total Time: ${med_tot}s"
        done
    done
done

# -----------------------------------------------------------------------------
# MPI + OPENMP
# -----------------------------------------------------------------------------
echo >> "$OUT"
echo "MPI+OMP" >> "$OUT"
echo "mode,n,nz,nodes,mpi_processes,threads_per_process,block_size,Total_Time_Med" >> "$OUT"

for mode in "${MODES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            echo
            echo ">> MPI+OpenMP: mode=$mode nodes=$MPI_NODES ranks=$MPI_PROCESSES threads/rank=$threads block=$block_size (repeats=$REPEATS)"

            med_tot=$(run_and_collect mpi_openmp "$mode" "$threads" "$block_size")

            echo "$mode,$N,$NZ,$MPI_NODES,$MPI_PROCESSES,$threads,$block_size,$med_tot" >> "$OUT"
            echo "mpi_openmp,$mode,$N,$NZ,$MPI_NODES,$MPI_PROCESSES,$threads,$block_size,$med_tot" >> "$FLAT_OUT"

            echo "  -> Median Total Time: ${med_tot}s"
        done
    done
done

echo
echo "Done. Sectioned results: $OUT"
echo "Done. Flat CSV results: $FLAT_OUT"
