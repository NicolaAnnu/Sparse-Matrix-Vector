// C++ threads implementation for the SPM One-Shot project.
// Uses the official usage() function provided by utils.hpp.
//
// This version preserves the algorithm of iterative_SpMV.cpp and uses a
// persistent thread pool. Work is split into chunks of -b logical rows.
//
// Synchronization does not rely on std::barrier:
//   - mutex + condition_variable coordinate the shared task queue;
//   - future::get() waits for each phase (SpMV, reduction, normalization).
//
// Build:
//   g++ -O3 -std=c++17 -pthread -I. -Wall -Wextra -Wpedantic
//       cpp_threads_SpMV.cpp -o cpp_threads

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "thread_pool.hpp"
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

static std::size_t chunk_count(std::size_t item_count, std::size_t chunk_size) {
    return item_count / chunk_size + (item_count % chunk_size != 0 ? 1 : 0);
}

template <class ChunkFunction>
static void parallel_for_chunks(ThreadPool& pool, std::size_t item_count, std::size_t chunk_size, ChunkFunction&& function) {
    const std::size_t number_of_chunks = chunk_count(item_count, chunk_size);
    
    std::vector<std::future<void>> completions;
    completions.reserve(number_of_chunks);

    for (std::size_t begin = 0; begin < item_count; begin += chunk_size) {
        const std::size_t end = std::min(begin + chunk_size, item_count);

        completions.emplace_back(pool.submit(
            [begin, end, &function] {
                function(begin, end);
            }));
    }

    // This loop is the phase synchronization point. If a worker task throws,
    // get() also propagates the exception to the main thread.
    for (std::future<void>& completion : completions) {
        completion.get();
    }
}

static double dot(const std::vector<double>& a, const std::vector<double>& b, ThreadPool& pool, std::size_t chunk_size) {
    
    const std::size_t number_of_chunks = chunk_count(a.size(), chunk_size);
    std::vector<std::future<double>> partial_results;
    partial_results.reserve(number_of_chunks);

    for (std::size_t begin = 0; begin < a.size(); begin += chunk_size) {
        const std::size_t end = std::min(begin + chunk_size, a.size());

        partial_results.emplace_back(pool.submit(
            [&a, &b, begin, end] {
                double partial = 0.0;
                for (std::size_t i = begin; i < end; ++i) {
                    partial += a[i] * b[i];
                }
                return partial;
            }));
    }

    // Futures are reduced in submission order. The order is deterministic for a
    // fixed chunk size, although it differs from the sequential element order.
    double total = 0.0;
    for (std::future<double>& partial : partial_results) {
        total += partial.get();
    }
    return total;
}

static void parallel_scale(std::vector<double>& vector, double factor, ThreadPool& pool, std::size_t chunk_size) {
    parallel_for_chunks(pool, vector.size(), chunk_size, [&vector, factor](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                vector[i] *= factor;
            }
        });
}

static void normalize(std::vector<double>& vector, ThreadPool& pool, std::size_t chunk_size) {
    const double norm = std::sqrt(dot(vector, vector, pool, chunk_size));
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        throw std::runtime_error("cannot normalize vector: invalid L2 norm");
    }

    parallel_scale(vector, 1.0 / norm, pool, chunk_size);
}

static void spmv_csr_shifted_rows(const CSRMatrix& matrix,
                                       std::size_t row_shift,
                                       const std::vector<double>& x,
                                       std::vector<double>& y,
                                       ThreadPool& pool,
                                       std::size_t chunk_size) {
    if (x.size() != matrix.n) {
        throw std::runtime_error("SpMV input vector has invalid size");
    }

    // Preserve the behavior of the sequential reference implementation.
    y.assign(matrix.n, 0.0);

    parallel_for_chunks(pool, matrix.n, chunk_size, [&matrix, row_shift, &x, &y](std::size_t begin, std::size_t end) {
            const std::size_t n = matrix.n;

            for (std::size_t logical_row = begin; logical_row < end; ++logical_row) {
                const std::size_t source_row =
                    (logical_row + n - row_shift) % n;

                double sum = 0.0;
                for (std::uint64_t position = matrix.row_ptr[source_row]; position < matrix.row_ptr[source_row + 1]; ++position) {
                    sum += matrix.values[position] * x[matrix.col_idx[position]];
                }

                y[logical_row] = sum;
            }
        });
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(const CSRMatrix& A,
                                               std::uint64_t seed,
                                               ThreadPool& pool,
                                               std::size_t chunk_size,
                                               std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // PHASE 1: initialize the vector used by the iterative method.
    // Parallel versions must preserve this initialization, or distribute the same initial vector, before entering the timed iterative loop.
    std::vector<double> x(n);
    std::vector<double> y(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }
    normalize(x, pool, chunk_size);

    // PHASE 2: iterative computation on the evolving matrix.
    // The sequential reference keeps the CSR matrix fixed and represents matrix
    // evolution through this logical row_shift value.
    std::size_t row_shift = 0;

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // At each epoch boundary, update the logical row mapping.
        if (iter > 0 && (iter % EPOCH_LEN) == 0) {
            row_shift = (row_shift + shift_rows) % n;
        }

        // One iteration: shifted SpMV followed by vector normalization.
        // The normalization contains a global reduction.
        spmv_csr_shifted_rows(A, row_shift, x, y, pool, chunk_size);
        normalize(y, pool, chunk_size);

        x.swap(y); // O(1), do not need to parallelize
    }

    // PHASE 3: final diagnostics for correctness checks.
    // The extra SpMV is used to compute the final Rayleigh-like value.
    spmv_csr_shifted_rows(A, row_shift, x, y, pool, chunk_size);
    const double rayleigh = dot(x, y, pool, chunk_size);
    const std::uint64_t checksum = checksum_vector(x);

    // Keep the final vector only if we have to dump it.
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

    std::cout << "SPARSE_ITERATION_THREADPOOL\n";
    std::cout
        << "threads=" << worker_count
        << "  block_rows=" << chunk_size << "\n";

    try {
        ThreadPool pool(worker_count);

        // Phase 1: input construction. 
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result = iterative_spmv_evolving(G.A, seed, pool, chunk_size, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

        // Phase 3: optional correctness support. Vector dumping is outside the timed region
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