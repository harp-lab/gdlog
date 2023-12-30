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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

int main(int argc, char *argv[]) {
    auto dataset_path = argv[1];
    int device_id;
    int number_of_sm;
    hipGetDevice(&device_id);
    hipDeviceGetAttribute(&number_of_sm, hipDeviceAttributeMultiprocessorCount,
                           device_id);
    int max_threads_per_block;
    hipDeviceGetAttribute(&max_threads_per_block,
                          hipDeviceAttributeMaxThreadsPerBlock, 0);
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

    int REPEAT = 10;
    // init the tuples
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "init tuples time: " << spent_time << std::endl;
    column_type *tuple_hashvs;
    hipMalloc((void **)&tuple_hashvs, graph_edge_counts * sizeof(column_type));
    column_type *col_tmp;
    hipMalloc((void **)&col_tmp, graph_edge_counts * sizeof(column_type));

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
    LIE tc_scc(grid_size, block_size);
    tc_scc.max_iteration = 277;
    tc_scc.reload_full_flag = false;
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tuple_generator_hook reorder_path_host;
    hipMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
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
    hipMalloc((void **)&dedup_buf,
               path_2__1_2->current_full_size * sizeof(tuple_type));
    hipDeviceSynchronize();
    tuple_type *dedup_buf_end = thrust::set_difference(
        thrust::device, path_2__1_2->newt->tuples,
        path_2__1_2->newt->tuples + path_2__1_2->newt->tuple_counts,
        path_2__1_2->tuple_full,
        path_2__1_2->tuple_full + path_2__1_2->current_full_size, dedup_buf,
        tuple_indexed_less(path_2__1_2->full->index_column_size,
                           path_2__1_2->full->arity -
                               path_2__1_2->dependent_column_size));
    tuple_size_t tp_counts = dedup_buf_end - dedup_buf;
    time_point_end = std::chrono::high_resolution_clock::now();
    spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                     time_point_end - time_point_begin)
                     .count();
    std::cout << "deduplicate with full time: " << spent_time << std::endl;

    // test merge speed
    
    tuple_type *merge_buf;
    std::cout << "start merge test ..." << std::endl;
    std::cout << "full size " << path_2__1_2->full->tuple_counts << std::endl;
    
    double alloc_time = 0;
    for (int i = 0; i < REPEAT; i++) {
        timer.start_timer();
        hipMalloc((void **)&merge_buf, (path_2__1_2->full->tuple_counts +
                                        tp_counts) * sizeof(tuple_type));
        timer.stop_timer();
        alloc_time += timer.get_spent_time();
        hipFree(merge_buf);
        merge_buf = nullptr;
    }
    hipMalloc((void **)&merge_buf, (path_2__1_2->full->tuple_counts +
                                        tp_counts) * sizeof(tuple_type));
    std::cout << "alloc merge buf time: " << alloc_time << std::endl;

    std::cout << "start merge test 2 ..." << std::endl;
    
    double resize_time = 0;
    for (int i = 0; i < REPEAT; i++) {
        thrust::device_vector<tuple_type> full_buf_vec(path_2__1_2->full->tuples, path_2__1_2->full->tuples + path_2__1_2->full->tuple_counts);
        timer.start_timer();
        full_buf_vec.resize(path_2__1_2->full->tuple_counts+ tp_counts);
        timer.stop_timer();
        resize_time += timer.get_spent_time();
    }
    
    std::cout << "resize merge buf time: " << resize_time << std::endl;
    
    std::cout << "dedup size " << tp_counts << std::endl;
    print_memory_usage();
    timer.start_timer();
    for (int i = 0; i < REPEAT; i++) {
        thrust::merge(thrust::device, path_2__1_2->tuple_full,
                      path_2__1_2->tuple_full + path_2__1_2->current_full_size,
                      dedup_buf, dedup_buf_end, merge_buf,
                      tuple_indexed_less(path_2__1_2->full->index_column_size,
                               path_2__1_2->full->arity));
    }
    timer.stop_timer();
    std::cout << "merge int once time: " << timer.get_spent_time() << std::endl;

    // std::cout << "start merge test 2 ..." << std::endl;
    // thrust::device_vector<tuple_type> full_buf_vec(path_2__1_2->full->tuples, path_2__1_2->full->tuples + path_2__1_2->full->tuple_counts);
    // thrust::device_vector<tuple_type> dedup_buf_vec(dedup_buf, dedup_buf_end);
    // for (int i = 0; i < REPEAT; i++) {
    //     timer.start_timer();
    //     thrust::merge(thrust::device, full_buf_vec.begin(),
    //                   full_buf_vec.end(),
    //                   dedup_buf_vec.begin(), dedup_buf_vec.end(), merge_buf,
    //                   tuple_indexed_less(path_2__1_2->full->index_column_size,
    //                            path_2__1_2->full->arity));
    //     timer.stop_timer();
    // }

    // std::cout << "start multi merge test ..." << std::endl;
    // tuple_size_t merge_step = 5000;
    // time_point_begin = std::chrono::high_resolution_clock::now();
    // for(tuple_size_t i = 0; i < path_2__1_2->full->tuple_counts; i += merge_step) {
    //     tuple_size_t merge_size = merge_step;
    //     if (i + merge_step > path_2__1_2->full->tuple_counts) {
    //         merge_size = path_2__1_2->full->tuple_counts - i;
    //     }
    //     hipDeviceSynchronize();
    //     thrust::merge(thrust::device, path_2__1_2->tuple_full + i,
    //                   path_2__1_2->tuple_full + i + merge_size,
    //                   dedup_buf, dedup_buf_end, merge_buf,
    //                   tuple_indexed_less(path_2__1_2->full->index_column_size,
    //                        path_2__1_2->full->arity));
    // }
    // hipDeviceSynchronize();
    // time_point_end = std::chrono::high_resolution_clock::now();
    // spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
    //                  time_point_end - time_point_begin)
    //                  .count();
    // std::cout << "multi merge time 1: " << spent_time << std::endl;

    // std::cout << "start multi merge test 2 ..." << std::endl;
    // timer.start_timer();
    // tuple_type *merge_buf_2;
    // hipMalloc((void **)&merge_buf_2, path_2__1_2->full->tuple_counts * sizeof(tuple_type));
    // tuple_type *merge_buf_3;
    // hipMalloc((void **)&merge_buf_3, path_2__1_2->full->tuple_counts * sizeof(tuple_type));
    // tuple_size_t cur_merged_size = 0;
    // print_memory_usage();
    // hipDeviceSynchronize();
    // time_point_begin = std::chrono::high_resolution_clock::now();
    // for(tuple_size_t i = 0; i < path_2__1_2->full->tuple_counts; i += merge_step) {
    //     tuple_size_t merge_size = merge_step;
    //     if (i + merge_step > path_2__1_2->full->tuple_counts) {
    //         merge_size = path_2__1_2->full->tuple_counts - i;
    //     }
    //     thrust::merge(thrust::device, path_2__1_2->tuple_full + i,
    //                   path_2__1_2->tuple_full + i + merge_size,
    //                   merge_buf_2, merge_buf_2 + cur_merged_size, merge_buf_2,
    //                   tuple_indexed_less(path_2__1_2->full->index_column_size,
    //                        path_2__1_2->full->arity));
    //     hipDeviceSynchronize();
    //     cur_merged_size += merge_size;
    // }
    // time_point_end = std::chrono::high_resolution_clock::now();
    // spent_time = std::chrono::duration_cast<std::chrono::duration<double>>(
    //                  time_point_end - time_point_begin)
    //                  .count();
    // std::cout << "multi merge time 2: " << spent_time << std::endl;
    hipFree(merge_buf);

    std::cout << "stupid test .... " << std::endl;
    thrust::host_vector<int> H1(4);
    // initialize individual elements
    H1[0] = 14;
    H1[1] = 20;
    H1[2] = 38;
    H1[3] = 46;
    thrust::host_vector<int> H2(3);
    // initialize individual elements
    H2[0] = 12;
    H2[1] = 31;
    H2[2] = 53;
    thrust::device_vector<int> h1_device = H1;
    thrust::device_vector<int> h2_device = H2;
    // h1_device.resize(7);
    thrust::merge(thrust::device, h1_device.begin(), h1_device.begin()+4,
                  h2_device.begin(), h2_device.end(), h1_device.begin(), thrust::less<int>());
    thrust::host_vector<int> H3 = h1_device;
    for (int i = 0; i < H3.size(); i++) {
        std::cout << H3[i] << std::endl;
    }

    return 0;
}
