#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <vector>

#include "../include/exception.h"
#include "../include/lie.h"
#include "../include/print.h"
#include "../include/timer.h"

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

__device__ void reorder_path(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = inner[1];
    newt[1] = outer[1];
};
__device__ tuple_generator_hook reorder_path_device = reorder_path;

void datastructure_bench(const char *dataset_path, int block_size,
                         int grid_size) {
    KernelTimer timer;
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
    column_type *raw_graph_data_gpu;
    hipMalloc((void **)&raw_graph_data_gpu,
               graph_edge_counts * 2 * sizeof(column_type));
    hipMemcpy(raw_graph_data_gpu, raw_graph_data,
               graph_edge_counts * 2 * sizeof(column_type),
               hipMemcpyHostToDevice);
    tuple_type *raw_graph_data_gpu_tuple;
    hipMalloc((void **)&raw_graph_data_gpu_tuple,
               graph_edge_counts * sizeof(tuple_type));
    init_tuples_unsorted<<<grid_size, block_size>>>(
        raw_graph_data_gpu_tuple, raw_graph_data_gpu, 2, graph_edge_counts);
    checkHip(hipDeviceSynchronize());
    thrust::sort(thrust::device, raw_graph_data_gpu_tuple,
                 raw_graph_data_gpu_tuple + graph_edge_counts,
                 tuple_indexed_less(1, 2));
    checkHip(hipDeviceSynchronize());
    column_type *raw_graph_data_gpu_sorted;
    hipMalloc((void **)&raw_graph_data_gpu_sorted,
               graph_edge_counts * 2 * sizeof(column_type));
    flatten_tuples_raw_data<<<grid_size, block_size>>>(
        raw_graph_data_gpu_tuple, raw_graph_data_gpu_sorted, graph_edge_counts,
        2);
    std::cout << "finish reverse graph." << std::endl;

    std::cout << "Testing datastructure build <<<<<<<<<<<<<<< " << std::endl;
    int REPEAT = 100;
    float build_table_time = 0;
    
    for (int i = 0; i < REPEAT; i++) {
        Relation *path_2__1_2 = new Relation();
        path_2__1_2->index_flag = false;
        // hipMallocHost((void **)&path_2__1_2, sizeof(Relation));
        // std::cout << "edge size " << graph_edge_counts << std::endl;
        // load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
        //             graph_edge_counts, 1, 0, grid_size, block_size);
        path_2__1_2->full = new GHashRelContainer(2, 1, 0);
        timer.start_timer();
        float load_detail_time[5] = {0, 0, 0, 0, 0};
        load_relation_container(path_2__1_2->full, 2, raw_graph_data_gpu_sorted,
                                graph_edge_counts, 1, 0, 0.8, grid_size,
                                block_size, load_detail_time, true, true);
        timer.stop_timer();
        build_table_time += timer.get_spent_time();
        // load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
        //               graph_edge_counts, 1, 0, grid_size, block_size);
        path_2__1_2->full->tuple_counts = 0;
        path_2__1_2->full->index_map_size = 0;
        path_2__1_2->full->data_raw_row_size = 0;
        if (path_2__1_2->full->index_map != nullptr) {
            checkHip(hipFree(path_2__1_2->full->index_map));
            path_2__1_2->full->index_map = nullptr;
        }
        if (path_2__1_2->full->tuples != nullptr) {
            checkHip(hipFree(path_2__1_2->full->tuples));
            path_2__1_2->full->tuples = nullptr;
        }
    }
    
    std::cout << "Graph size: " << graph_edge_counts << std::endl;
    std::cout << "Build hash table time: " << build_table_time << std::endl;
    std::cout << "HashTable build ratio : "
              << graph_edge_counts * REPEAT / build_table_time << std::endl;

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
    tuple_generator_hook reorder_path_host;
    hipMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));
    std::cout << "Testing datastructure query <<<<<<<<<<<<<<< " << std::endl;
    
    float join_time[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    RelationalJoin join_test(edge_2__2_1, FULL, path_2__1_2, FULL, path_2__1_2,
                             reorder_path_host, nullptr, LEFT, grid_size,
                             block_size, join_time);
    float query_time = 0;
    for (int i = 0; i < REPEAT; i++) {
        join_test.disable_load = true;
        timer.start_timer();
        join_test();
        timer.stop_timer();
        query_time += timer.get_spent_time();
    }

    std::cout << "Query time: " << query_time << std::endl;
    std::cout << "HashTable query ratio : "
              << graph_edge_counts * REPEAT / query_time << std::endl;
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    hipGetDevice(&device_id);
    hipDeviceGetAttribute(&number_of_sm, hipDeviceAttributeMultiprocessorCount,
                           device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    datastructure_bench(argv[1], block_size, grid_size);
    return 0;
}
