#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <vector>

#define EMPTY_HASH_ENTRY ULLONG_MAX

using u64 = unsigned long long;
using u32 = unsigned long;

using column_type = u32;
using tuple_type = column_type *;
using tuple_size_t = u64;
using t_data_internal = u64 *;

typedef void (*tuple_generator_hook)(tuple_type, tuple_type, tuple_type);
typedef void (*tuple_copy_hook)(tuple_type, tuple_type);
typedef bool (*tuple_predicate)(tuple_type);

// struct tuple_generator_hook {
//     __host__ __device__
//     void operator()(tuple_type inner, tuple_type outer, tuple_type newt) {};
// };

/**
 * @brief TODO: remove this use comparator function
 *
 * @param t1
 * @param t2
 * @param l
 * @return true
 * @return false
 */
__host__ __device__ inline bool tuple_eq(tuple_type t1, tuple_type t2,
                                         tuple_size_t l) {
    for (int i = 0; i < l; i++) {
        if (t1[i] != t2[i]) {
            return false;
        }
    }
    return true;
}

struct t_equal {
    u64 arity;

    t_equal(tuple_size_t arity) { this->arity = arity; }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {
        for (int i = 0; i < arity; i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief fnv1-a hash used in original slog backend
 *
 * @param start_ptr
 * @param prefix_len
 * @return __host__ __device__
 */
__host__ __device__ inline u64 prefix_hash(tuple_type start_ptr,
                                           u64 prefix_len) {
    const u64 base = 14695981039346656037ULL;
    const u64 prime = 1099511628211ULL;

    u64 hash = base;
    for (u64 i = 0; i < prefix_len; ++i) {
        u64 chunk = (u64)start_ptr[i];
        hash ^= chunk & 255ULL;
        hash *= prime;
        for (char j = 0; j < 7; ++j) {
            chunk = chunk >> 8;
            hash ^= chunk & 255ULL;
            hash *= prime;
        }
    }
    return hash;
}

// 32 bit version of fnv1-a
__host__ __device__ inline u32 prefix_hash_32(tuple_type start_ptr,
                                              u64 prefix_len) {
    const u32 base = 2166136261U;
    const u32 prime = 16777619U;

    u32 hash = base;
    for (u64 i = 0; i < prefix_len; ++i) {
        u32 chunk = (u32)start_ptr[i];
        hash ^= chunk & 255U;
        hash *= prime;
        for (char j = 0; j < 3; ++j) {
            chunk = chunk >> 8;
            hash ^= chunk & 255U;
            hash *= prime;
        }
    }
    return hash;
}

// 32bit xxhash version prefix hash
__host__ __device__ inline u32 prefix_hash_xxhash_32(tuple_type start_ptr,
                                                     u64 prefix_len) {
    const u32 prime = 2654435761U;
    u32 hash = 0;
    for (u64 i = 0; i < prefix_len; ++i) {
        u32 chunk = (u32)start_ptr[i];
        hash += chunk * prime;
        hash += (hash << 13);
        hash ^= (hash >> 7);
        hash += (hash << 3);
        hash ^= (hash >> 17);
        hash += (hash << 5);
    }
    return hash;
}

// change to std
struct tuple_indexed_less {

    // u64 *index_columns;
    tuple_size_t index_column_size;
    int arity;

    tuple_indexed_less(tuple_size_t index_column_size, int arity) {
        // this->index_columns = index_columns;
        this->index_column_size = index_column_size;
        this->arity = arity;
    }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {
        // fetch the index
        // compare hash first, could be index very different but share the same
        // hash
        if (prefix_hash(lhs, index_column_size) ==
            prefix_hash(rhs, index_column_size)) {
            // same hash
            for (tuple_size_t i = 0; i < arity; i++) {
                if (lhs[i] < rhs[i]) {
                    return true;
                } else if (lhs[i] > rhs[i]) {
                    return false;
                }
            }
            return false;
        } else if (prefix_hash_xxhash_32(lhs, index_column_size) <
                   prefix_hash_xxhash_32(rhs, index_column_size)) {
            return true;
        } else {
            return false;
        }
    }
};

struct tuple_weak_less {

    int arity;

    tuple_weak_less(int arity) { this->arity = arity; }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {

        for (u64 i = 0; i < arity; i++) {
            if (lhs[i] < rhs[i]) {
                return true;
            } else if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    };
};

long int get_row_size(const char *data_path) {
    std::ifstream f;
    f.open(data_path);
    char c;
    long i = 0;
    while (f.get(c))
        if (c == '\n')
            ++i;
    f.close();
    return i;
}

enum ColumnT { U64, U32 };

column_type *get_relation_from_file(const char *file_path, int total_rows,
                                    int total_columns, char separator,
                                    ColumnT ct) {
    column_type *data =
        (column_type *)malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                if (ct == U64) {
                    fscanf(data_file, "%lld%c", &data[(i * total_columns) + j],
                           &separator);
                } else {
                    fscanf(data_file, "%ld%c", &data[(i * total_columns) + j],
                           &separator);
                }
            } else {
                if (ct == U64) {
                    fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
                } else {
                    fscanf(data_file, "%ld", &data[(i * total_columns) + j]);
                }
            }
        }
    }
    return data;
}

// print tuples
void print_tuple_list(tuple_type *tuples, tuple_size_t rows,
                      tuple_size_t arity) {
    tuple_type *tuples_host;
    cudaMallocHost((void **)&tuples_host, rows * sizeof(tuple_type));
    cudaMemcpy(tuples_host, tuples, rows * sizeof(tuple_type),
               cudaMemcpyDeviceToHost);
    if (rows > 100) {
        rows = 100;
    }
    for (tuple_size_t i = 0; i < rows; i++) {
        tuple_type cur_tuple = tuples_host[i];

        tuple_type cur_tuple_host;
        cudaMallocHost((void **)&cur_tuple_host, arity * sizeof(column_type));
        cudaMemcpy(cur_tuple_host, cur_tuple, arity * sizeof(column_type),
                   cudaMemcpyDeviceToHost);
        for (tuple_size_t j = 0; j < arity; j++) {
            std::cout << cur_tuple_host[j] << " ";
        }
        std::cout << std::endl;
    }
}

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// Number of bits per pass
const int BITS_PER_PASS = 4;

// Number of bins per pass
const int BINS_PER_PASS = 1 << BITS_PER_PASS;

// Number of threads per block
const int THREADS_PER_BLOCK = 256;

// Radix sort kernel
__global__ void radix_sort_kernel(u32 *data, int *temp, int *histogram,
                                  int num_elements, int pass) {
    // Compute the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the local thread ID within the warp
    int lane = threadIdx.x & 31;

    // Compute the histogram index for this thread
    int index = (data[tid] >> (pass * BITS_PER_PASS)) & (BINS_PER_PASS - 1);

    // Compute the starting index for this bin in the temp array
    int start = histogram[index * blockDim.x + lane];

    // Compute the ending index for this bin in the temp array
    int end = start + histogram[index * blockDim.x + blockDim.x - 1];

    // Copy the element to the temp array
    temp[start + lane] = data[tid];

    // Increment the histogram count for this bin
    atomicAdd(&histogram[index * blockDim.x + lane], 1);

    // Wait for all threads to finish updating the histogram
    __syncthreads();

    // Compute the starting index for this thread's bin in the temp array
    start = histogram[index * blockDim.x + lane];

    // Copy the element to the temp array
    temp[start + lane] = data[tid];

    // Wait for all threads to finish copying to the temp array
    __syncthreads();

    // Update the data array with the sorted elements
    data[tid] = temp[tid];
}

// Radix sort function
void radix_sort(column_type *data, int arity, int num_elements) {
    // Allocate memory for the temp array and histogram
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    int *temp, *histogram;
    cudaMalloc(&temp, num_elements * sizeof(int));
    cudaMalloc(&histogram, BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));

    // Initialize the histogram to zero
    cudaMemset(histogram, 0, BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));
    column_type pass_cnt = sizeof(column_type) * 8 * arity / BITS_PER_PASS;

    // Perform the radix sort passes
    for (column_type pass = 0; pass < pass_cnt; pass++) {
        // Launch the radix sort kernel
        radix_sort_kernel<<<(num_elements + THREADS_PER_BLOCK - 1) /
                                THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK>>>(data+arity, temp, histogram,
                                                 num_elements, pass);

        // Clear the histogram for the next pass
        cudaMemset(histogram, 0,
                   BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));
    }

    // Free the memory
    cudaFree(temp);
    cudaFree(histogram);
}

__global__ void init_tuples_unsorted(tuple_type *tuples, column_type *raw_data,
                                     int arity, tuple_size_t rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        tuples[i] = raw_data + i * arity;
    }
}

// cuda kernel compute hash for tuples using fnv1-a
__global__ void compute_hash(tuple_type *tuples, tuple_size_t rows,
                             tuple_size_t index_column_size,
                             column_type *hashes) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        hashes[i] =
            (column_type)prefix_hash_xxhash_32(tuples[i], index_column_size);
    }
}

// cuda kernel extract the k th column from tuples
__global__ void extract_column(tuple_type *tuples, tuple_size_t rows,
                               tuple_size_t k, column_type *column) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        column[i] = tuples[i][k];
    }
}

int main(int argc, char *argv[]) {
    auto dataset_path = argv[1];
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    std::cout << "num of sm " << number_of_sm << " num of thread per block " << max_threads_per_block << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = std::chrono::high_resolution_clock::now();
    double spent_time;

    // load the raw graph
    tuple_size_t graph_edge_counts = get_row_size(dataset_path);
    std::cout << "Input graph rows: " << graph_edge_counts << std::endl;
    // u64 graph_edge_counts = 2100;
    column_type *raw_graph_data =
        get_relation_from_file(dataset_path, graph_edge_counts, 2, '\t', U32);
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));
    std::cout << "reversing graph ... " << std::endl;
    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    std::cout << "finish reverse graph." << std::endl;

    // copy the graph to device
    column_type *d_graph_data;
    cudaMalloc((void **)&d_graph_data,
               graph_edge_counts * relation_columns * sizeof(column_type));
    cudaMemcpy(d_graph_data, raw_graph_data,
               graph_edge_counts * relation_columns * sizeof(column_type),
               cudaMemcpyHostToDevice);

    int REPEAT = 1;
    // init the tuples
    tuple_type *tuples;
    cudaMalloc(&tuples, graph_edge_counts * sizeof(tuple_type));
    time_point_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPEAT; i++) {
        init_tuples_unsorted<<<grid_size, block_size>>>(
            tuples, d_graph_data, relation_columns, graph_edge_counts);
    }
    cudaDeviceSynchronize();
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "init tuples time: " << spent_time << std::endl;
    column_type *tuple_hashvs;
    cudaMalloc((void **)&tuple_hashvs, graph_edge_counts * sizeof(column_type));
    column_type *col_tmp;
    cudaMalloc((void **)&col_tmp, graph_edge_counts * sizeof(column_type));

    time_point_end = std::chrono::high_resolution_clock::now();
    // compute hash for tuples
    for (int i = 0; i < REPEAT; i++) {
        compute_hash<<<grid_size, block_size>>>(tuples, graph_edge_counts, 1,
                                                tuple_hashvs);
        cudaDeviceSynchronize();
    }
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "compute hash time: " << spent_time << std::endl;

    // sort the tuples using thrust
    double sort_hash_time = 0;
    for (int i = 0; i < REPEAT; i++) {
        time_point_begin = std::chrono::high_resolution_clock::now();

        extract_column<<<grid_size, block_size>>>(tuples, graph_edge_counts, 1,
                                                  col_tmp);
        cudaDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, col_tmp,
                                   col_tmp + graph_edge_counts, tuples);
        cudaDeviceSynchronize();
        extract_column<<<grid_size, block_size>>>(tuples, graph_edge_counts, 0,
                                                  col_tmp);
        cudaDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, col_tmp,
                                   col_tmp + graph_edge_counts, tuples);
        compute_hash<<<grid_size, block_size>>>(tuples, graph_edge_counts, 1,
                                                tuple_hashvs);
        cudaDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, tuple_hashvs,
                                   tuple_hashvs + graph_edge_counts, tuples);
        cudaDeviceSynchronize();
        time_point_end = std::chrono::high_resolution_clock::now();
        sort_hash_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                time_point_end - time_point_begin)
                .count();
        print_tuple_list(tuples, graph_edge_counts, 2);
        // recover prepare for next sort
        init_tuples_unsorted<<<grid_size, block_size>>>(
            tuples, d_graph_data, relation_columns, graph_edge_counts);
    }
    std::cout << "sort hash time: " << sort_hash_time << std::endl;

    // sort the tuples using thrust with tuple_indexed_less
    double sort_comp_time = 0;
    for (int i = 0; i < REPEAT; i++) {
        time_point_begin = std::chrono::high_resolution_clock::now();
        thrust::sort(thrust::device, tuples, tuples + graph_edge_counts,
                     tuple_indexed_less(1, 2));
        cudaDeviceSynchronize();
        time_point_end = std::chrono::high_resolution_clock::now();
        sort_comp_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                time_point_end - time_point_begin)
                .count();
        print_tuple_list(tuples, graph_edge_counts, 2);
        init_tuples_unsorted<<<grid_size, block_size>>>(
            tuples, d_graph_data, relation_columns, graph_edge_counts);
    }
    std::cout << "sort using tuple_indexed_less time: " << sort_comp_time
              << std::endl;

    return 0;
}
