#include <chrono>
#include <hip/hip_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <vector>

#include "../include/relation.h"
#include "../include/timer.h"
#include "../include/relational_algebra.h"
#include "../include/print.h"

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
    hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, 0);
    int *temp, *histogram;
    hipMalloc(&temp, num_elements * sizeof(int));
    hipMalloc(&histogram, BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));

    // Initialize the histogram to zero
    hipMemset(histogram, 0, BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));
    column_type pass_cnt = sizeof(column_type) * 8 * arity / BITS_PER_PASS;

    // Perform the radix sort passes
    for (column_type pass = 0; pass < pass_cnt; pass++) {
        // Launch the radix sort kernel
        radix_sort_kernel<<<(num_elements + THREADS_PER_BLOCK - 1) /
                                THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK>>>(data+arity, temp, histogram,
                                                 num_elements, pass);

        // Clear the histogram for the next pass
        hipMemset(histogram, 0,
                   BINS_PER_PASS * THREADS_PER_BLOCK * sizeof(int));
    }

    // Free the memory
    hipFree(temp);
    hipFree(histogram);
}

struct t_equal_n {
    u64 arity;
    tuple_type rhs;

    t_equal_n(tuple_size_t arity, tuple_type target) { this->arity = arity; this->rhs = target; }

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
    hipGetDevice(&device_id);
    hipDeviceGetAttribute(&number_of_sm, hipDeviceAttributeMultiprocessorCount,
                           device_id);
    int max_threads_per_block;
    hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, 0);
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
    hipMalloc((void **)&d_graph_data,
               graph_edge_counts * relation_columns * sizeof(column_type));
    hipMemcpy(d_graph_data, raw_graph_data,
               graph_edge_counts * relation_columns * sizeof(column_type),
               hipMemcpyHostToDevice);

    int REPEAT = 1;
    // init the tuples
    tuple_type *tuples;
    hipMalloc(&tuples, graph_edge_counts * sizeof(tuple_type));
    time_point_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPEAT; i++) {
        init_tuples_unsorted<<<grid_size, block_size>>>(
            tuples, d_graph_data, relation_columns, graph_edge_counts);
    }
    hipDeviceSynchronize();
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "init tuples time: " << spent_time << std::endl;
    column_type *tuple_hashvs;
    hipMalloc((void **)&tuple_hashvs, graph_edge_counts * sizeof(column_type));
    column_type *col_tmp;
    hipMalloc((void **)&col_tmp, graph_edge_counts * sizeof(column_type));

    time_point_end = std::chrono::high_resolution_clock::now();
    // compute hash for tuples
    for (int i = 0; i < REPEAT; i++) {
        compute_hash<<<grid_size, block_size>>>(tuples, graph_edge_counts, 1,
                                                tuple_hashvs);
        hipDeviceSynchronize();
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
        hipDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, col_tmp,
                                   col_tmp + graph_edge_counts, tuples);
        hipDeviceSynchronize();
        extract_column<<<grid_size, block_size>>>(tuples, graph_edge_counts, 0,
                                                  col_tmp);
        hipDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, col_tmp,
                                   col_tmp + graph_edge_counts, tuples);
        compute_hash<<<grid_size, block_size>>>(tuples, graph_edge_counts, 1,
                                                tuple_hashvs);
        hipDeviceSynchronize();
        thrust::stable_sort_by_key(thrust::device, tuple_hashvs,
                                   tuple_hashvs + graph_edge_counts, tuples);
        hipDeviceSynchronize();
        time_point_end = std::chrono::high_resolution_clock::now();
        sort_hash_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                time_point_end - time_point_begin)
                .count();
        // print_tuple_list(tuples, graph_edge_counts, 2);
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
        hipDeviceSynchronize();
        time_point_end = std::chrono::high_resolution_clock::now();
        sort_comp_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                time_point_end - time_point_begin)
                .count();
        // print_tuple_list(tuples, graph_edge_counts, 2);
        init_tuples_unsorted<<<grid_size, block_size>>>(
            tuples, d_graph_data, relation_columns, graph_edge_counts);
    }
    std::cout << "sort using tuple_indexed_less time: " << sort_comp_time
              << std::endl;


    // load raw data into edge relation
    time_point_begin = std::chrono::high_resolution_clock::now();
    Relation *edge_2__2_1 = new Relation();
    // hipMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    Relation *path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // hipMallocHost((void **)&path_2__1_2, sizeof(Relation));
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    time_point_end = std::chrono::high_resolution_clock::now();
    // double kernel_spent_time = timer.get_spent_time();
    double init_relation_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            time_point_end - time_point_begin)
            .count();
    std::cout << "Build hash table time: " << init_relation_time << std::endl;

    tuple_generator_hook reorder_path_host;
    hipMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    RelationalJoin join_test(edge_2__2_1, FULL, path_2__1_2, FULL, path_2__1_2,
                             reorder_path_host, nullptr, LEFT, grid_size,
                             block_size, join_detail);
    time_point_begin = std::chrono::high_resolution_clock::now();
    join_test();
    time_point_end = std::chrono::high_resolution_clock::now();
    double join_test_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            time_point_end - time_point_begin)
            .count();
    std::cout << "join test time: " << join_test_time << std::endl;
    std::cout << "join detail: " << std::endl;
    std::cout << "compute size time:  " <<  join_detail[0] <<  std::endl;
    std::cout << "reduce + scan time: " <<  join_detail[1] <<  std::endl;
    std::cout << "fetch result time:  " <<  join_detail[2] <<  std::endl;
    std::cout << "sort time:          " <<  join_detail[3] <<  std::endl;
    std::cout << "build index time:   " <<  join_detail[5] <<  std::endl;
    std::cout << "merge time:         " <<  join_detail[6] <<  std::endl;
    std::cout << "unique time:        " << join_detail[4] + join_detail[7] <<  std::endl;
    // test thrust set_difference time on path's newt and full
    tuple_type* deduped_tuples;
    hipMalloc(&deduped_tuples, path_2__1_2->newt->tuple_counts * sizeof(tuple_type));

    time_point_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
    thrust::set_difference(thrust::device, path_2__1_2->newt->tuples,
                           path_2__1_2->newt->tuples + path_2__1_2->newt->tuple_counts,
                           path_2__1_2->full->tuples, path_2__1_2->full->tuples + path_2__1_2->full->tuple_counts,
                           deduped_tuples, tuple_indexed_less(1, 2));
    hipDeviceSynchronize();
    }
    time_point_end = std::chrono::high_resolution_clock::now();
    double set_difference_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            time_point_end - time_point_begin)
            .count();
    std::cout << "set_difference time: " << set_difference_time << std::endl;

    // sequential set_difference
    tuple_type* deduped_tuples_seq;
    hipMalloc(&deduped_tuples_seq, path_2__1_2->newt->tuple_counts * sizeof(tuple_type));
    time_point_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        tuple_type* full_t_end = path_2__1_2->full->tuples + path_2__1_2->full->tuple_counts;
        for (auto i = 0; i < path_2__1_2->newt->tuple_counts ; i++) {
            auto cur_newt_tuple = path_2__1_2->newt->tuples[i];
            
            auto res =thrust::find_if(thrust::device, path_2__1_2->full->tuples, path_2__1_2->full->tuples + path_2__1_2->full->tuple_counts,
                            t_equal_n(path_2__1_2->arity, cur_newt_tuple));
            hipDeviceSynchronize();
            if (res != full_t_end) {
                deduped_tuples_seq[i] = cur_newt_tuple;
            }
        }
    }
    time_point_end = std::chrono::high_resolution_clock::now();
    double set_difference_time_seq =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            time_point_end - time_point_begin)
            .count();
    std::cout << "set_difference time seq: " << set_difference_time_seq << std::endl;

    return 0;
}
