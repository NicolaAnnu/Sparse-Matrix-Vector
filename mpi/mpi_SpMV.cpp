// Hybrid MPI + OpenMP implementation of the Iterative Sparse Matrix-Vector Computation
//
// Command line:
//   mpirun -np P ./mpi -n N -nz K -m mode -t T -b B
//
// Minimal build:
//   mpic++ -O3 -std=c++17 -fopenmp -I. -Wall -Wextra -Wpedantic mpi_SpMV.cpp -o mpi
//
// Examples:
//   salloc -N 2
//   mpirun -n 2 ./mpi -n 500000 -nz 20000000 -m regular -t 4 -b 1024
//   mpirun -n 4 ./mpi -n 500000 -nz 20000000 -m irregular -t 2 -b 256
//   mpirun -n 4 ./mpi -n 5000 -nz 20000 -m irregular -t 2 -b 256 --dump-vector mpi_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time.
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
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "matrix_generation.hpp"
#include "utils.hpp"

static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t shift = n / 16 + 17;
    if ((shift % 2) == 0) ++shift;
    shift %= n;
    if (shift == 0) shift = 1;
    return shift;
}

struct alignas(64) PaddedDouble {
    double value = 0.0;
};

// Same static partitioning used in the OpenMP implementation.
static double dot(const std::vector<double>& a,
                  const std::vector<double>& b,
                  std::size_t worker_count) {
    if (a.size() != b.size())
        throw std::runtime_error("dot: incompatible vector sizes");
    if (a.empty()) return 0.0;

    worker_count = std::min(worker_count, a.size());
    std::vector<PaddedDouble> partial(worker_count);

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = a.size() * worker / worker_count;
        const std::size_t end   = a.size() * (worker + 1) / worker_count;

        #pragma omp task default(none) shared(a, b, partial) firstprivate(worker, begin, end)
        {
            double local = 0.0;
            for (std::size_t i = begin; i < end; ++i)
                local += a[i] * b[i];
            partial[worker].value = local;
        }
    }

    #pragma omp taskwait

    double total = 0.0;
    for (const auto& p : partial) total += p.value;
    return total;
}

static void parallel_scale(std::vector<double>& x,
                           double factor,
                           std::size_t worker_count) {
    if (x.empty()) return;

    worker_count = std::min(worker_count, x.size());
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = x.size() * worker / worker_count;
        const std::size_t end   = x.size() * (worker + 1) / worker_count;

        #pragma omp task default(none) shared(x) firstprivate(factor, begin, end)
        {
            for (std::size_t i = begin; i < end; ++i)
                x[i] *= factor;
        }
    }

    #pragma omp taskwait
}

static void normalize(std::vector<double>& x, std::size_t worker_count) {
    const double norm = std::sqrt(dot(x, x, worker_count));
    parallel_scale(x, 1.0 / norm, worker_count);
}

// SpMV on the rows owned by the MPI process.
static double local_spmv_csr(
    const std::vector<std::uint64_t>& row_ptr,
    const std::vector<std::uint32_t>& col_idx,
    const std::vector<double>& values,
    const std::vector<double>& x,
    std::vector<double>& y,
    std::size_t chunk_size)
{
    const std::size_t local_n = y.size();
    const std::size_t chunks =
        (local_n + chunk_size - 1) / chunk_size;

    std::vector<PaddedDouble> partial_norm(chunks);

    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t begin = chunk * chunk_size;
        const std::size_t end =
            std::min(begin + chunk_size, local_n);

        #pragma omp task default(none) \
            shared(row_ptr, col_idx, values, x, y, partial_norm) \
            firstprivate(begin, end, chunk)
        {
            double local_norm = 0.0;

            for (std::size_t row = begin; row < end; ++row) {
                double sum = 0.0;

                for (std::uint64_t p = row_ptr[row];
                     p < row_ptr[row + 1];
                     ++p) {

                    sum += values[p] * x[col_idx[p]];
                }

                y[row] = sum;
                local_norm += sum * sum;
            }

            partial_norm[chunk].value = local_norm;
        }
    }

    #pragma omp taskwait

    double norm2 = 0.0;

    for (const auto& p : partial_norm)
        norm2 += p.value;

    return norm2;
}

// Source row i becomes logical row (i + row_shift) mod n.
static void rotate_vector(const std::vector<double>& source,
                          std::vector<double>& logical,
                          std::size_t row_shift,
                          std::size_t chunk_size) {
    const std::size_t n = source.size();
    const std::size_t chunks = (n + chunk_size - 1) / chunk_size;

    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t begin = chunk * chunk_size;
        const std::size_t end   = std::min(begin + chunk_size, n);

        #pragma omp task default(none) shared(source, logical) firstprivate(begin, end, row_shift, n)
        {
            for (std::size_t i = begin; i < end; ++i)
                logical[(i + row_shift) % n] = source[i];
        }
    }

    #pragma omp taskwait
}

struct IterativeResult {
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;
    std::size_t final_row_shift = 0;
    double distribution_time = 0.0;
    double local_computation_time = 0.0;
    double communication_time = 0.0;
    double reduction_time = 0.0;
    double epoch_time = 0.0;
};

static IterativeResult distributed_iterative_spmv(
    int rank, int num_ranks,
    const CSRMatrix& global_A,
    std::uint64_t nz,
    std::size_t n,
    std::uint64_t seed,
    std::size_t worker_count,
    std::size_t chunk_size,
    std::vector<double>* final_vector = nullptr) {

    const std::size_t shift_rows = compute_shift_rows(n);

    // Phase 1: distribute contiguous row ranges, approximately balanced by NNZ.
    const auto td0 = std::chrono::steady_clock::now();

    std::vector<int> row_counts(num_ranks), row_displacements(num_ranks);

    if (rank == 0) {
        std::vector<std::size_t> boundary(num_ranks + 1, 0);
        boundary[num_ranks] = n;

        for (int r = 1; r < num_ranks; ++r) {
            const std::uint64_t target = (nz / num_ranks) * r;
            auto it = std::lower_bound(global_A.row_ptr.begin(),
                                       global_A.row_ptr.end(), target);
            std::size_t b = static_cast<std::size_t>(it - global_A.row_ptr.begin());
            b = std::max(b, boundary[r - 1] + 1);
            b = std::min(b, n - static_cast<std::size_t>(num_ranks - r));
            boundary[r] = b;
        }

        for (int r = 0; r < num_ranks; ++r) {
            row_displacements[r] = static_cast<int>(boundary[r]);
            row_counts[r] = static_cast<int>(boundary[r + 1] - boundary[r]);
        }
    }

    MPI_Bcast(row_counts.data(), num_ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_displacements.data(), num_ranks, MPI_INT, 0, MPI_COMM_WORLD);

    const int local_n = row_counts[rank];
    std::vector<int> nnz_counts(num_ranks), nnz_displacements(num_ranks);
    std::vector<int> ptr_counts(num_ranks), ptr_displacements(num_ranks);

    if (rank == 0) {
        for (int r = 0; r < num_ranks; ++r) {
            const std::size_t first = row_displacements[r];
            const std::size_t last  = first + row_counts[r];
            nnz_displacements[r] = static_cast<int>(global_A.row_ptr[first]);
            nnz_counts[r] = static_cast<int>(global_A.row_ptr[last] - global_A.row_ptr[first]);
            ptr_displacements[r] = row_displacements[r];
            ptr_counts[r] = row_counts[r] + 1;
        }
    }

    int local_nnz = 0;
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT,
                &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> local_values(local_nnz);
    std::vector<std::uint32_t> local_col_idx(local_nnz);
    std::vector<std::uint64_t> local_row_ptr(local_n + 1);

    MPI_Scatterv(rank == 0 ? global_A.values.data() : nullptr,
                 nnz_counts.data(), nnz_displacements.data(), MPI_DOUBLE,
                 local_values.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? global_A.col_idx.data() : nullptr,
                 nnz_counts.data(), nnz_displacements.data(), MPI_UINT32_T,
                 local_col_idx.data(), local_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? global_A.row_ptr.data() : nullptr,
                 ptr_counts.data(), ptr_displacements.data(), MPI_UINT64_T,
                 local_row_ptr.data(), local_n + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    const std::uint64_t offset = local_row_ptr[0];
    for (auto& p : local_row_ptr) p -= offset;

    const auto td1 = std::chrono::steady_clock::now();
    double distribution_time = std::chrono::duration<double>(td1 - td0).count();

    // Phase 2: same iterative structure as the OpenMP implementation.
    std::vector<double> x(n), y(n), global_out(n);
    std::vector<double> local_out(local_n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) v = rng.next_unit();

    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;
    double local_computation_time = 0.0;
    double communication_time = 0.0;
    double reduction_time = 0.0;
    double epoch_time = 0.0;

    #pragma omp parallel num_threads(worker_count) shared(x, y, global_out, local_out, row_shift, rayleigh, checksum)
    {
        // The OpenMP master is also the only thread that calls MPI.
        #pragma omp master
        {
            auto t0 = std::chrono::steady_clock::now();
            normalize(x, worker_count);
            auto t1 = std::chrono::steady_clock::now();
            local_computation_time += std::chrono::duration<double>(t1 - t0).count();

            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    t0 = std::chrono::steady_clock::now();
                    row_shift = (row_shift + shift_rows) % n;
                    t1 = std::chrono::steady_clock::now();
                    epoch_time += std::chrono::duration<double>(t1 - t0).count();
                }

                t0 = std::chrono::steady_clock::now();
                const double local_norm2 =
                local_spmv_csr(
                    local_row_ptr,
                    local_col_idx,
                    local_values,
                    x,
                    local_out,
                    chunk_size);
                t1 = std::chrono::steady_clock::now();
                local_computation_time += std::chrono::duration<double>(t1 - t0).count();

                double global_norm2 = 0.0;
                t0 = std::chrono::steady_clock::now();
                MPI_Allreduce(&local_norm2, &global_norm2, 1,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                t1 = std::chrono::steady_clock::now();
                reduction_time += std::chrono::duration<double>(t1 - t0).count();

                t0 = std::chrono::steady_clock::now();
                parallel_scale(local_out, 1.0 / std::sqrt(global_norm2), worker_count);
                t1 = std::chrono::steady_clock::now();
                local_computation_time += std::chrono::duration<double>(t1 - t0).count();

                t0 = std::chrono::steady_clock::now();
                MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE,
                               global_out.data(), row_counts.data(),
                               row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                t1 = std::chrono::steady_clock::now();
                communication_time += std::chrono::duration<double>(t1 - t0).count();

                t0 = std::chrono::steady_clock::now();
                rotate_vector(global_out, y, row_shift, chunk_size);
                x.swap(y);
                t1 = std::chrono::steady_clock::now();
                local_computation_time += std::chrono::duration<double>(t1 - t0).count();
            }

            // Phase 3: final diagnostics, as in the OpenMP version.
            local_spmv_csr(local_row_ptr, local_col_idx, local_values,
                           x, local_out, chunk_size);
            MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE,
                           global_out.data(), row_counts.data(),
                           row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            rotate_vector(global_out, y, row_shift, chunk_size);
            rayleigh = dot(x, y, worker_count);
            checksum = checksum_vector(x);
        }
    }

    double global_distribution = 0.0;
    double global_local_computation = 0.0;
    double global_communication = 0.0;
    double global_reduction = 0.0;
    double global_epoch = 0.0;

    MPI_Reduce(&distribution_time, &global_distribution, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_computation_time, &global_local_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication_time, &global_communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&reduction_time, &global_reduction, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&epoch_time, &global_epoch, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (final_vector != nullptr && rank == 0)
        *final_vector = std::move(x);

    return IterativeResult{
        rayleigh,
        checksum,
        row_shift,
        global_distribution,
        global_local_computation,
        global_communication,
        global_reduction,
        global_epoch
    };
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI_THREAD_FUNNELED not supported\n";
        MPI_Finalize();
        return 1;
    }

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Phase 0: same input pattern as sequential/OpenMP.
    std::uint64_t n64 = 0;
    std::uint64_t nz = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads = 0;
    std::uint64_t block_size64 = 1024;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads)) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_u64(argc, argv, "-b", block_size64);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    const std::size_t worker_count = static_cast<std::size_t>(num_threads);
    const std::size_t chunk_size = static_cast<std::size_t>(block_size64);

    if (worker_count == 0 || chunk_size == 0 || chunk_size > n ||
        static_cast<std::size_t>(num_ranks) > n) {
        if (rank == 0) std::cerr << "Error: invalid MPI/OpenMP configuration\n";
        MPI_Finalize();
        return 1;
    }

    try {
        // Phase 1: matrix generation only on the master process.
        GeneratedMatrix G;
        double generation_sec = 0.0;

        if (rank == 0) {
            const auto tg0 = std::chrono::steady_clock::now();
            G = generate_matrix(n, nz, seed, mode);
            const auto tg1 = std::chrono::steady_clock::now();
            generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

            print_matrix_stats(G);
            std::cout << "generation_time_sec=" << generation_sec << "\n\n";
        }

        std::vector<double> final_vector;
        std::vector<double>* final_vector_out =
            dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed distributed computation.
        MPI_Barrier(MPI_COMM_WORLD);
        const auto tc0 = std::chrono::steady_clock::now();

        const IterativeResult result = distributed_iterative_spmv(
            rank, num_ranks, G.A, nz, n, seed,
            worker_count, chunk_size, final_vector_out);

        MPI_Barrier(MPI_COMM_WORLD);
        const auto tc1 = std::chrono::steady_clock::now();

        if (rank == 0) {
            const double computation_sec =
                std::chrono::duration<double>(tc1 - tc0).count();

            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << computation_sec << "\n";
            std::cout << "distribution_time=" << result.distribution_time << "\n";
            std::cout << "local_computation_time=" << result.local_computation_time << "\n";
            std::cout << "communication_time=" << result.communication_time << "\n";
            std::cout << "global_reduction_time=" << result.reduction_time << "\n";
            std::cout << "epoch_transition_time=" << result.epoch_time << "\n";

            // Phase 3: optional dump outside the timed region.
            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}