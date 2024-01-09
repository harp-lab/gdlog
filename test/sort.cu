#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <vector>

#include "../include/print.cuh"
#include "../include/relation.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"
#include "../include/exception.cuh"

#define EMPTY_HASH_ENTRY ULLONG_MAX
#define REPEAT 10

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
    cudaDeviceGetAttribute(&max_threads_per_block,
                           cudaDevAttrMaxThreadsPerBlock, 0);
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
                            THREADS_PER_BLOCK>>>(data + arity, temp, histogram,
                                                 num_elements, pass);

        // Clear the histogram for the next pass
        cudaMemset(histogram, 0,
                   BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));
    }

    // Free the memory
    cudaFree(temp);
    cudaFree(histogram);
}

struct t_equal_n {
    u64 arity;
    tuple_type rhs;

    t_equal_n(tuple_size_t arity, tuple_type target) {
        this->arity = arity;
        this->rhs = target;
    }

    __host__ __device__ bool operator()(tuple_type lhs) {
        for (int i = 0; i < arity; i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

__device__ void reorder_path(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = inner[1];
    newt[1] = outer[1];
};
__device__ tuple_generator_hook reorder_path_device = reorder_path;

int main(int argc, char *argv[]) {
    auto dataset_path = argv[1];
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block,
                           cudaDevAttrMaxThreadsPerBlock, 0);
    std::cout << "num of sm " << number_of_sm << " num of thread per block "
              << max_threads_per_block << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;
    ;
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
    
    return 0;
}
