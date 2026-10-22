# Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices


[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-Tasking%20%26%20Worksharing-green.svg)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-PMIx%20%2F%20Slurm-orange.svg)](https://www.mpi-forum.org/)
[![Python](https://img.shields.io/badge/Python-3-yellow.svg)](https://www.python.org/)

This project studies iterative sparse matrix-vector multiplication (SpMV) on a matrix stored in CSR format. The computation performs **500 iterations**, with a logical circular row shift every **25 iterations**. The matrix is not physically reordered when an epoch changes.

The project contains five implementations:

| Implementation | Source file | Approach |
|---|---|---|
| Sequential | `sequential/iterative_SpMV.cpp` | Reference implementation |
| C++ Threads | `thread/thread_SpMV.cpp` | Persistent ThreadPool; dynamic SpMV chunks |
| OpenMP tasks | `openmp/openmp_SpMV.cpp` | Explicit tasks for SpMV chunks |
| OpenMP work-sharing | `openmp/openmp_SpMV_worksharing.cpp` | Dynamic `omp for` for SpMV; static loops for vector operations |
| MPI + OpenMP | `mpi/mpi_SpMV.cpp` | NNZ-balanced MPI partitions and local OpenMP tasks |

The implementation and experimental results are described in `REPORT_SPM..pdf`.

## Requirements

- A C++17 compiler (`g++`), with OpenMP and POSIX threads support.
- An MPI installation with `mpic++` and a Slurm cluster with `srun` for distributed benchmarks.
- Python 3 with `pandas` and `matplotlib` **only** for regenerating plots.

## Compile

From the extracted **project root** on the SPM cluster:

```bash
g++ -O3 -std=c++17 -pthread -I. sequential/iterative_SpMV.cpp -o seq
g++ -O3 -std=c++17 -pthread -I. thread/thread_SpMV.cpp -o thread_SpMV
g++ -O3 -std=c++17 -fopenmp -I. openmp/openmp_SpMV.cpp -o openmp_SpMV
g++ -O3 -std=c++17 -fopenmp -I. openmp/openmp_SpMV_worksharing.cpp -o openmp_SpMV_worksharing
mpic++ -O3 -std=c++17 -fopenmp -I. mpi/mpi_SpMV.cpp -o mpi/mpi
```

**Important:** the MPI executable is `mpi/mpi` because `mpi/` is a source directory. Before running the MPI experiment scripts, update the executable path in their common configuration:

```bash
sed -i 's|MPI_BIN="../mpi"|MPI_BIN="../mpi/mpi"|' experiments/common_config_mpi.sh
```

## Run the programs

Common parameters are `-n N` (matrix dimension), `-nz K` (total nonzeros), `-m regular|irregular`, and optional `-s SEED` (default `111`). Parallel implementations also accept `-t THREADS` and `-b BLOCK_SIZE` (rows per chunk). `--dump-vector FILE` saves the final vector for correctness checks.

Run benchmarks **on compute nodes using Slurm**, not directly on the login node. Example: one-node OpenMP run using four physical cores:

```bash
srun --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --hint=nomultithread --time=00:05:00 \
  env OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./openmp_SpMV -n 5000 -nz 20000 -m irregular -s 111 -t 4 -b 256
```

Example: MPI + OpenMP with two ranks and two threads per rank on one compute node:

```bash
srun --partition=normal --nodes=1 --ntasks=2 --ntasks-per-node=2 \
  --cpus-per-task=2 --hint=nomultithread --mpi=pmix --time=00:05:00 \
  env OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
  ./mpi/mpi -n 5000 -nz 20000 -m irregular -s 111 -t 2 -b 256
```

All programs print the execution time, Rayleigh value and checksum. Matrix generation has its own timer and is **excluded** from computation time.

## CPU affinity and thread placement

The benchmarks **use CPU affinity**, configured in `experiments/common_config.sh`, `common_config_openmp.sh`, and `common_config_mpi.sh`.

- Each cluster node has **16 physical cores / 32 logical CPUs**. For up to 16 threads, Slurm requests `--cpus-per-task=THREADS` with `--hint=nomultithread`, avoiding SMT siblings.
- OpenMP and MPI + OpenMP use `OMP_PLACES=cores` and `OMP_PROC_BIND=close` to bind threads close to one another. The MPI launcher also sets `OMP_DYNAMIC=FALSE`.
- For **32-thread** OpenMP/MPI configurations, the launchers use `--cpu-bind=threads` and `OMP_PLACES=threads` to include logical CPUs (SMT).
- The C++ ThreadPool benchmarks receive CPU allocations through Slurm rather than using OpenMP affinity variables.
- The MPI rank/thread sweep retains `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, and `--hint=nomultithread`; it deliberately does **not** add `--cpu-bind=cores`.

## Correctness checks

Compare each parallel version with the sequential reference using the **same** `N`, `NZ`, mode and seed. The following small tests generate final vectors for an element-wise comparison (run from the project root):

```bash
srun --partition=normal -N 1 -n 1 -c 1 --time=00:05:00 \
  ./seq -n 1000 -nz 1000 -m irregular -s 111 --dump-vector seq.dump

srun --partition=normal -N 1 -n 1 -c 4 --hint=nomultithread --time=00:05:00 \
  ./thread_SpMV -n 1000 -nz 1000 -m irregular -s 111 -t 4 -b 256 --dump-vector thread.dump

srun --partition=normal -N 1 -n 1 -c 4 --hint=nomultithread --time=00:05:00 \
  env OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./openmp_SpMV -n 1000 -nz 1000 -m irregular -s 111 -t 4 -b 256 --dump-vector openmp.dump

srun --partition=normal -N 1 -n 1 -c 4 --hint=nomultithread --time=00:05:00 \
  env OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./openmp_SpMV_worksharing -n 1000 -nz 1000 -m irregular -s 111 -t 4 -b 256 --dump-vector worksharing.dump

srun --partition=normal -N 1 -n 2 --ntasks-per-node=2 -c 2 \
  --hint=nomultithread --mpi=pmix --time=00:05:00 \
  env OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
  ./mpi/mpi -n 1000 -nz 1000 -m irregular -s 111 -t 2 -b 256 --dump-vector mpi.dump
```

Compare the generated vectors with Python (no third-party packages required):

```bash
python3 - <<'PY'
import math

with open('seq.dump') as f:
    reference = [float(line) for line in f]

for name in ('thread', 'openmp', 'worksharing', 'mpi'):
    with open(f'{name}.dump') as f:
        result = [float(line) for line in f]
    passed = len(reference) == len(result) and all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        for a, b in zip(reference, result)
    )
    print(f'{name}: {"PASS" if passed else "FAIL"}')
PY
```

The Rayleigh value provides an additional consistency check, including on larger inputs. Checksums may differ due to floating-point operation order and are **not** required to match bit-for-bit.

## Benchmark scripts

The scripts are in `experiments/` and use Slurm to request cluster resources. **Run them from that directory**, because they load the local `common_config*.sh` files:

```bash
cd experiments

# Shared-memory experiments
bash THREAD_run_baseline_speedup.sh
bash THREAD_run_thread_scaling.sh
bash THREAD_run_task_granularity.sh
bash OPENMP_run_strong_scaling.sh
bash OPENMP_run_task_granularity.sh
bash OPENMP_run_task_vs_worksharing.sh

# Distributed experiments
bash MPI_run_task_granularity.sh
bash MPI_run_strong_scaling.sh
bash MPI_run_weak_scaling.sh
bash MPI_run_rank_thread_sweep.sh
bash MPI_run_sweep_balance.sh

# Regular versus irregular workloads
bash ALL_run_regular_vs_irregular.sh
```

Each configuration is repeated **three times** and the scripts record median timings in `experiments/results/` as CSV files. The main reference workload uses `N=1,000,000`, `NZ=200,000,000`, `irregular`, seed `111`. The tuned MPI configuration uses one MPI rank per node, 16 OpenMP threads per rank, and block size `2048`. Strong scaling uses 1, 2, 4, 6, 8 nodes; weak scaling increases both `N` and `NZ` proportionally with the number of nodes.

To regenerate plots, first ensure all required CSV files exist (in particular, `ALL_run_regular_vs_irregular.sh` creates `regular_vs_irregular_all_flat.csv`), then run:

```bash
python3 zplot_results.py
```

Figures are saved in `experiments/plots/`. Some full-size experiments require up to eight compute nodes and may take substantial cluster resources.
