# Iterative SpMV

Four implementations of an iterative sparse matrix-vector computation with
periodic logical row shifts: sequential reference, C++ threads (custom
ThreadPool), OpenMP (task-based and work-sharing), and MPI+OpenMP.

See `REPORT_SPM..pdf` for implementation strategies, correctness methodology,
and the full experimental analysis.

## Project structure

```text
sequential/   sequential reference (iterative_SpMV.cpp)
thread/       C++ threads version, custom ThreadPool (thread_SpMV.cpp)
openmp/       OpenMP task-based (openmp_SpMV.cpp) and work-sharing
              (openmp_SpMV_worksharing.cpp) versions
mpi/          MPI+OpenMP distributed version (mpi_SpMV.cpp)
experiments/  scripts, CSV results, and plots for the experimental analysis
```

## Build

From the project root, execute the following commands on a compute node:

```bash
g++ -O3 -std=c++20 sequential/iterative_SpMV.cpp -o seq
g++ -O3 -std=c++20 -pthread thread/thread_SpMV.cpp -o thread_SpMV
g++ -O3 -std=c++20 -fopenmp openmp/openmp_SpMV.cpp -o openmp_SpMV
g++ -O3 -std=c++20 -fopenmp openmp/openmp_SpMV_worksharing.cpp -o openmp_SpMV_worksharing
mpic++ -O3 -std=c++20 -fopenmp mpi/mpi_SpMV.cpp -o mpi/mpi
```

Requires a C++20 compiler with OpenMP support and an MPI implementation
providing `mpic++`. Executables are placed in the project root, except for
the MPI executable, which is placed inside `mpi/`.

## Running

All executables share the same core arguments:

```text
-n N        matrix size, N x N
-nz K       total number of nonzeros
-m mode     regular | irregular
-s seed     optional, default 111
```

Parallel versions additionally take `-t` (worker threads, per rank for MPI)
and `-b` (rows per chunk, default 1024). MPI+OpenMP is launched with `mpirun`:

```bash
./seq -n 500000 -nz 20000000 -m irregular
./thread_SpMV -n 500000 -nz 20000000 -m irregular -t 8 -b 1024
./openmp_SpMV -n 500000 -nz 20000000 -m irregular -t 8 -b 1024
./openmp_SpMV_worksharing -n 500000 -nz 20000000 -m irregular -t 8 -b 1024
mpirun -n 4 ./mpi/mpi -n 500000 -nz 20000000 -m irregular -t 4 -b 1024
```

Each run prints the Rayleigh value, a checksum, and execution time.
MPI+OpenMP also reports distribution, SpMV, normalization, rotation,
communication, global reduction, and epoch-transition timings.

## Correctness checks

Compare each implementation against the sequential reference using identical
inputs and seed. Pass `--dump-vector FILE` to dump the final normalized
vector outside the timed region:

```bash
./seq -n 5000 -nz 20000 -m irregular --dump-vector seq.dump
mpirun -n 2 ./mpi/mpi -n 5000 -nz 20000 -m irregular -t 2 -b 1024 --dump-vector mpi.dump
```

For a strong check, compare the dumped vectors element by element within a
numerical tolerance. For larger inputs, comparing the Rayleigh values provides
a weaker consistency check. Parallel floating-point reduction order can
produce small differences, so bitwise-identical vectors or checksums are
not required.

## Reproducing the experiments

Run the `*_run_*.sh` scripts from `experiments/` on the spmcluster login node;
they request resources through `srun`. Set `MPI_BIN="../mpi/mpi"` in
`common_config_mpi.sh` to match the build above, and review the shared
configuration files and experiment parameters before running.

Experiments cover scaling, speedup, task granularity, task-based versus
work-sharing OpenMP, regular versus irregular matrices, and MPI rank/thread
balance. Some scripts reuse earlier CSV results. `zplot_results.py` generates
plots using pandas and Matplotlib; its `regular_vs_irregular_all_flat.csv`
input is produced by `ALL_run_regular_vs_irregular.sh`.
