// OpenMP version of the Sparse Matrix-Vector multiplication.
//
// Parallelization strategy derived from the original C++ ThreadPool version:
//   - one persistent OpenMP parallel region is reused across all iterations;
//   - one thread (omp single) orchestrates the iterative algorithm and creates tasks;
//   - dot product and vector scaling preserve the original static partitioning;
//   - SpMV creates one OpenMP task per chunk, replacing the manual atomic
//     next_chunk scheduler used by the ThreadPool implementation;
//   - #pragma omp taskwait replaces the future.get() synchronization phases.
//
// Command line:
//   -n  N          matrix size, NxN
//   -nz K          total number of nonzeros
//   -m  mode       regular | irregular
//   -s  seed       optional seed, default 111
//   -t  T          number of OpenMP threads
//   -b  B          number of logical rows contained in an SpMV chunk,
//                  default 1024
//   --dump-vector FILE
//                  optional dump of the final normalized vector
//
// Minimal build:
//   g++ -O3 -std=c++17 -fopenmp -I. -Wall -Wextra -Wpedantic openmp_SpMV.cpp -o openmp_spmv
//
// Examples:
//   ./openmp_spmv -n 500000 -nz 20000000 -m regular -t 2
//   ./openmp_spmv -n 500000 -nz 20000000 -m irregular -t 4 -b 1024
//   ./openmp_spmv -n 5000 -nz 20000 -m irregular -t 2 -b 1024 --dump-vector omp_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time and has its own timer.
//   - The OpenMP team is created inside iterative_spmv_evolving().
//   - Dot product and vector scaling use static partitions, one task per partition.
//   - SpMV uses task-based chunk scheduling managed by the OpenMP runtime.
//   - The computation uses a fixed number of iterations.
//   - The main workload is the irregular case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"

static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t shift = n / 16 + 17;
    if ((shift % 2) == 0) {
        ++shift;
    }
    shift %= n;
    if (shift == 0) {
        shift = 1;
    }
    return shift;
}

struct alignas(64) PaddedDouble {
    double value = 0.0;
};

// Static partitioning, as in the original ThreadPool implementation.
// This function must be called from inside an OpenMP parallel region.
static double dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::size_t worker_count)
{
    if (a.size() != b.size()) {
        throw std::runtime_error("dot: incompatible vector sizes");
    }

    worker_count = std::min(worker_count, a.size());
    std::vector<PaddedDouble> partial(worker_count);

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = a.size() * worker / worker_count;
        const std::size_t end   = a.size() * (worker + 1) / worker_count;

        #pragma omp task default(none) shared(a, b, partial) firstprivate(worker, begin, end)
        {
            double local = 0.0;

            for (std::size_t i = begin; i < end; ++i) {
                local += a[i] * b[i];
            }

            partial[worker].value = local;
        }
    }

    // Replaces waiting on all futures in the ThreadPool implementation.
    #pragma omp taskwait

    double total = 0.0;
    for (const auto& value : partial) {
        total += value.value;
    }

    return total;
}

// Static partitioning, as in parallel_for_workers() from the ThreadPool version.
// This function must be called from inside an OpenMP parallel region.
static void parallel_scale(
    std::vector<double>& vector,
    double factor,
    std::size_t worker_count)
{
    worker_count = std::min(worker_count, vector.size());

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = vector.size() * worker / worker_count;
        const std::size_t end   = vector.size() * (worker + 1) / worker_count;

        #pragma omp task default(none) shared(vector) firstprivate(factor, begin, end)
        {
            for (std::size_t i = begin; i < end; ++i) {
                vector[i] *= factor;
            }
        }
    }

    #pragma omp taskwait
}

// This function must be called from inside an OpenMP parallel region.
static void normalize(
    std::vector<double>& vector,
    std::size_t worker_count)
{
    const double norm =
        std::sqrt(dot(vector, vector, worker_count));

    parallel_scale(
        vector,
        1.0 / norm,
        worker_count);
}

// Task-based SpMV.
// The original ThreadPool version launched worker_count long-lived jobs and
// used an atomic next_chunk counter. Here we instead create one OpenMP task
// for every chunk and let the OpenMP runtime dynamically assign those tasks
// to the available threads.
// This function must be called from inside an OpenMP parallel region.
static void spmv_csr_shifted_rows(
    const CSRMatrix& matrix,
    std::size_t row_shift,
    const std::vector<double>& x,
    std::vector<double>& y,
    std::size_t chunk_size)
{
    if (x.size() != matrix.n) {
        throw std::runtime_error("SpMV input vector has invalid size");
    }

    if (y.size() != matrix.n) {
        y.resize(matrix.n);
    }

    const std::size_t number_of_chunks =
        (matrix.n + chunk_size - 1) / chunk_size;

    for (std::size_t chunk = 0; chunk < number_of_chunks; ++chunk) {
        const std::size_t begin = chunk * chunk_size;
        const std::size_t end   = std::min(begin + chunk_size, matrix.n);

        #pragma omp task default(none) shared(matrix, x, y) firstprivate(row_shift, begin, end)
        {
            for (std::size_t logical_row = begin;
                 logical_row < end;
                 ++logical_row) {

                const std::size_t source_row =
                    (logical_row + matrix.n - row_shift) % matrix.n;

                double sum = 0.0;

                for (std::uint64_t position = matrix.row_ptr[source_row];
                     position < matrix.row_ptr[source_row + 1];
                     ++position) {

                    sum += matrix.values[position] *
                           x[matrix.col_idx[position]];
                }

                y[logical_row] = sum;
            }
        }
    }

    // The single/orchestrating task cannot continue to normalization until
    // every SpMV chunk for this iteration has completed.
    #pragma omp taskwait
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(
    const CSRMatrix& A,
    std::uint64_t seed,
    std::size_t worker_count,
    std::size_t chunk_size,
    std::vector<double>* final_vector = nullptr)
{
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // PHASE 1: preserve exactly the same sequential RNG initialization.
    std::vector<double> x(n);
    std::vector<double> y(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }

    // Values produced inside the OpenMP region and returned afterwards.
    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;

    // Persistent OpenMP team: created once and reused for normalization,
    // every SpMV iteration, and the final diagnostics.
    #pragma omp parallel num_threads(worker_count) shared(x, y, row_shift, rayleigh, checksum)
    {
        // A single thread orchestrates the sequential control flow and creates
        // tasks. All threads in the team are allowed to execute those tasks.
        #pragma omp single
        {
            // The initialized vector must be normalized before iteration 0.
            normalize(x, worker_count);

            // PHASE 2: iterative computation on the evolving matrix.
            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }

                spmv_csr_shifted_rows(
                    A,
                    row_shift,
                    x,
                    y,
                    chunk_size);

                normalize(y, worker_count);

                // Both SpMV and normalize() end with taskwait, therefore no
                // task still accesses x/y here.
                x.swap(y);
            }

            // PHASE 3: final diagnostics for correctness checks.
            spmv_csr_shifted_rows(
                A,
                row_shift,
                x,
                y,
                chunk_size);

            rayleigh = dot(x, y, worker_count);

            // Preserve the checksum implementation used by the original code.
            // It is executed by the single orchestrating thread.
            checksum = checksum_vector(x);
        }
    }

    if (final_vector != nullptr) {
        *final_vector = std::move(x);
    }

    return IterativeResult{
        rayleigh,
        checksum,
        row_shift
    };
}

int main(int argc, char** argv) {
    // Phase 0: read problem size, sparsity mode, seed, and optional dump path.
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::string mode;
    std::string dump_vector_path;
    std::uint64_t block_size64 = 1024;
    std::uint64_t num_threads = 0;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads)) {
        usage(argv[0]);
        return 1;
    }

    // Optional arguments
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_u64(argc, argv, "-b", block_size64);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    const std::size_t worker_count = static_cast<std::size_t>(num_threads);
    const std::size_t chunk_size = static_cast<std::size_t>(block_size64);

    std::cout << "SPARSE_ITERATION_OPENMP\n";
    std::cout
        << "threads=" << worker_count
        << "  block_rows=" << chunk_size << "\n";

    try {
        // Phase 1: input construction.
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec =
            std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double> final_vector;
        std::vector<double>* final_vector_out =
            dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result = iterative_spmv_evolving(
            G.A,
            seed,
            worker_count,
            chunk_size,
            final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec =
            std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum
                  << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

        // Phase 3: optional correctness support. Vector dumping is outside
        // the timed region.
        if (!dump_vector_path.empty()) {
            dump_vector(dump_vector_path, final_vector);
            std::cout << "vector_dump=" << dump_vector_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}