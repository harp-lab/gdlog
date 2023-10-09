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

#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

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
    KernelTimer timer;

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

    int REPEAT = 1;
    // init the tuples
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "init tuples time: " << spent_time << std::endl;
    column_type *tuple_hashvs;
    cudaMalloc((void **)&tuple_hashvs, graph_edge_counts * sizeof(column_type));
    column_type *col_tmp;
    cudaMalloc((void **)&col_tmp, graph_edge_counts * sizeof(column_type));

    // load raw data into edge relation
    time_point_begin = std::chrono::high_resolution_clock::now();
    Relation *edge_2__2_1 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    Relation *path_2__1_2 = new Relation();
    path_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    LIE tc_scc(grid_size, block_size);
    tc_scc.max_iteration = 1;
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tuple_generator_hook reorder_path_host;
    cudaMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));
    tuple_copy_hook cp_1_host;
    RelationalJoin join_op(edge_2__2_1, FULL, path_2__1_2, DELTA, path_2__1_2,
                           reorder_path_host, nullptr, LEFT, grid_size,
                           block_size, join_detail);
    tc_scc.add_ra(join_op);
    timer.start_timer();
    tc_scc.fixpoint_loop();
    timer.stop_timer();
    std::cout << "Path counts " << path_2__1_2->full->tuple_counts << std::endl;
    // print_tuple_rows(path_2__2_1->full, "full");
    std::cout << "TC time: " << timer.get_spent_time() << std::endl;
    std::cout << "join detail: " << std::endl;
    std::cout << "compute size time:  " << join_detail[0] << std::endl;
    std::cout << "reduce + scan time: " << join_detail[1] << std::endl;
    std::cout << "fetch result time:  " << join_detail[2] << std::endl;
    std::cout << "sort time:          " << join_detail[3] << std::endl;
    std::cout << "build index time:   " << join_detail[5] << std::endl;
    std::cout << "merge time:         " << join_detail[6] << std::endl;
    std::cout << "unique time:        " << join_detail[4] + join_detail[7]
              << std::endl;

    join_op();
    print_memory_usage();
    // deduplicate with full
    time_point_begin = std::chrono::high_resolution_clock::now();
    std::cout << "start deduplicate with full ..." << std::endl;
    tuple_type *dedup_buf;
    cudaMalloc((void **)&dedup_buf,
               path_2__1_2->current_full_size * sizeof(tuple_type));
    cudaDeviceSynchronize();
    tuple_type *dedup_buf_end = thrust::set_difference(
        thrust::device, path_2__1_2->newt->tuples,
        path_2__1_2->newt->tuples + path_2__1_2->newt->tuple_counts,
        path_2__1_2->tuple_full,
        path_2__1_2->tuple_full + path_2__1_2->current_full_size, dedup_buf,
        tuple_indexed_less(path_2__1_2->full->index_column_size,
                           path_2__1_2->full->arity -
                               path_2__1_2->dependent_column_size));
    cudaDeviceSynchronize();
    tuple_size_t tp_counts = dedup_buf_end - dedup_buf;
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "deduplicate with full time: " << spent_time << std::endl;

    // test merge speed
    time_point_begin = std::chrono::high_resolution_clock::now();
    tuple_type *merge_buf;
    std::cout << "start merge test ..." << std::endl;
    timer.start_timer();
    cudaMalloc((void **)&merge_buf, path_2__1_2->full->tuple_counts +
                                        tp_counts * sizeof(tuple_type));
    print_memory_usage();
    cudaDeviceSynchronize();
    thrust::merge(thrust::device, path_2__1_2->tuple_full,
                  path_2__1_2->tuple_full + path_2__1_2->current_full_size,
                  dedup_buf, dedup_buf_end, merge_buf);
    timer.stop_timer();
    std::cout << "merge int once time: " << timer.get_spent_time() << std::endl;

    std::cout << "start multi merge test ..." << std::endl;
    tuple_size_t merge_step = 5000;
    time_point_begin = std::chrono::high_resolution_clock::now();
    for(tuple_size_t i = 0; i < path_2__1_2->full->tuple_counts; i += merge_step) {
        tuple_size_t merge_size = merge_step;
        if (i + merge_step > path_2__1_2->full->tuple_counts) {
            merge_size = path_2__1_2->full->tuple_counts - i;
        }
        cudaDeviceSynchronize();
        thrust::merge(thrust::device, path_2__1_2->tuple_full + i,
                      path_2__1_2->tuple_full + i + merge_size,
                      dedup_buf, dedup_buf_end, merge_buf);
    }
    cudaDeviceSynchronize();
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "multi merge time: " << spent_time << std::endl;
    cudaFree(merge_buf);
    return 0;
}
