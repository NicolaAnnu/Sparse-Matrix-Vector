// OpenMP WORK-SHARING version of the Sparse Matrix-Vector multiplication.
//
// Parallelization strategy derived from the original C++ ThreadPool version:
//   - one persistent OpenMP parallel region is reused across all iterations;
//   - all threads cooperate on the iterative computation;
//   - dot product and vector scaling use static work-sharing;
//   - SpMV uses dynamic work-sharing to balance the irregular workload;
//   - implicit OpenMP barriers synchronize the different computation phases.
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
//   g++ -O3 -std=c++17 -fopenmp -I. -Wall -Wextra -Wpedantic openmp_SpMV_worksharing.cpp -o openmp_SpMV_worksharing
//
// Examples:
//   ./openmp_SpMV_worksharing -n 500000 -nz 20000000 -m regular -t 2
//   ./openmp_SpMV_worksharing -n 500000 -nz 20000000 -m irregular -t 4 -b 1024
//   ./openmp_SpMV_worksharing -n 5000 -nz 20000 -m irregular -t 2 -b 1024 --dump-vector omp_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time and has its own timer.
//   - The OpenMP team is created inside iterative_spmv_evolving().
//   - Dot product and vector scaling use static work-sharing.
//   - SpMV uses dynamic work-sharing managed by the OpenMP runtime.
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

#include <omp.h>

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

// Static work-sharing.
// This function must be called by all threads from inside an OpenMP parallel region.
static void dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<PaddedDouble>& partial,
    double& total)
{
    const std::size_t thread_id =
        static_cast<std::size_t>(omp_get_thread_num());

    double local = 0.0;

    #pragma omp for schedule(static) nowait
    for (std::size_t i = 0; i < a.size(); ++i) {
        local += a[i] * b[i];
    }

    partial[thread_id].value = local;

    #pragma omp barrier

    #pragma omp single
    {
        total = 0.0;

        for (const auto& value : partial) {
            total += value.value;
        }
    }
}

// Static work-sharing.
// This function must be called by all threads from inside an OpenMP parallel region.
static void parallel_scale(
    std::vector<double>& vector,
    double factor)
{
    #pragma omp for schedule(static)
    for (std::size_t i = 0; i < vector.size(); ++i) {
        vector[i] *= factor;
    }
}

// This function must be called by all threads from inside an OpenMP parallel region.
static void normalize(
    std::vector<double>& vector,
    std::vector<PaddedDouble>& partial,
    double& norm)
{
    dot(
        vector,
        vector,
        partial,
        norm);

    #pragma omp single
    {
        norm = std::sqrt(norm);
    }

    const double factor = 1.0 / norm;

    parallel_scale(
        vector,
        factor);
}

// Work-sharing SpMV.
// Rows are distributed dynamically among OpenMP threads.
// Dynamic scheduling is useful for the irregular workload because
// different rows may contain very different numbers of nonzeros.
// This function must be called by all threads from inside an OpenMP parallel region.
static void spmv_csr_shifted_rows(
    const CSRMatrix& matrix,
    std::size_t row_shift,
    const std::vector<double>& x,
    std::vector<double>& y,
    std::size_t chunk_size)
{
    #pragma omp for schedule(dynamic, chunk_size)
    for (std::size_t logical_row = 0;
         logical_row < matrix.n;
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
    double norm = 0.0;
    std::uint64_t checksum = 0;

    // One partial result per OpenMP thread.
    std::vector<PaddedDouble> partial(worker_count);

    // Persistent OpenMP team: created once and reused for normalization,
    // every SpMV iteration, and the final diagnostics.
    #pragma omp parallel num_threads(worker_count) default(none) \
        shared(A, x, y, row_shift, rayleigh, norm, checksum, partial, \
               shift_rows, n, chunk_size)
    {
        // The initialized vector must be normalized before iteration 0.
        normalize(
            x,
            partial,
            norm);

        // PHASE 2: iterative computation on the evolving matrix.
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {

            // The row shift changes only at epoch boundaries.
            if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                #pragma omp single
                {
                    row_shift = (row_shift + shift_rows) % n;
                }
            }

            // All threads cooperate on the SpMV through dynamic work-sharing.
            spmv_csr_shifted_rows(
                A,
                row_shift,
                x,
                y,
                chunk_size);

            // All threads cooperate on normalization.
            normalize(
                y,
                partial,
                norm);

            // Only one thread swaps the shared vectors.
            #pragma omp single
            {
                x.swap(y);
            }
        }

        // PHASE 3: final diagnostics for correctness checks.
        spmv_csr_shifted_rows(
            A,
            row_shift,
            x,
            y,
            chunk_size);

        dot(
            x,
            y,
            partial,
            rayleigh);

        // Preserve the checksum implementation used by the task-based code.
        // It is executed by one thread.
        #pragma omp single
        {
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

    std::cout << "SPARSE_ITERATION_OPENMP_WORKSHARING\n";
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