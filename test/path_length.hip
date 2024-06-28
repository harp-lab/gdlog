#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.h"
#include "../include/lie.h"
#include "../include/print.h"
#include "../include/timer.h"

//////////////////////////////////////////////////////

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

column_type *get_relation_from_file(const char *file_path, int total_rows,
                                    int total_columns, char separator) {
    column_type *data =
        (column_type *)malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%lld%c", &data[(i * total_columns) + j],
                       &separator);
            } else {
                fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
            }
        }
    }
    return data;
}

//////////////////////////////////////////////////////////////////

__device__ void reorder_path(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = inner[1];
    newt[1] = outer[1];
    newt[2] = outer[2] + 1;
};
__device__ tuple_generator_hook reorder_path_device = reorder_path;

void analysis_bench(const char *dataset_path, int block_size, int grid_size) {
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = std::chrono::high_resolution_clock::now();
    double spent_time;

    // load the raw graph
    u64 graph_edge_counts = get_row_size(dataset_path);
    std::cout << "Input graph rows: " << graph_edge_counts << std::endl;
    // u64 graph_edge_counts = 2100;
    column_type *raw_graph_data =
        get_relation_from_file(dataset_path, graph_edge_counts, 2, '\t');
    column_type *raw_reverse_graph_data;
    u64 raw_reverse_graph_data_mem_size =
        graph_edge_counts * 2 * sizeof(column_type);
    hipMallocHost((void **)&raw_reverse_graph_data,
                   raw_reverse_graph_data_mem_size);
    hipMemset(raw_reverse_graph_data, 0, raw_reverse_graph_data_mem_size);
    column_type *raw_path_data;
    u64 raw_path_data_mem_size = graph_edge_counts * 3 * sizeof(column_type);
    hipMallocHost((void **)&raw_path_data,
                   raw_path_data_mem_size);
    hipMemset(raw_path_data, 0, raw_path_data_mem_size);

    std::cout << "init path ... " << std::endl;
    for (u64 i = 0; i < graph_edge_counts; i++) {
        raw_path_data[i * 3] = raw_graph_data[i * 2];
        raw_path_data[i * 3 + 1] = raw_graph_data[i * 2 + 1];
        raw_path_data[i * 3 + 2] = 1;
    }

    std::cout << "reversing graph ... " << std::endl;
    for (u64 i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    std::cout << "finish reverse graph." << std::endl;

    timer.start_timer();
    Relation *edge_2__2_1 = new Relation();
    // hipMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    Relation *path_3__1_2_3 = new Relation();
    path_3__1_2_3->index_flag = false;
    // hipMallocHost((void **)&path_3__1_2_3, sizeof(Relation));
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(path_3__1_2_3, "path_3__1_2_3", 3, raw_path_data,
                  graph_edge_counts, 1, 1, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    timer.stop_timer();
    // double kernel_spent_time = timer.get_spent_time();
    std::cout << "Build hash table time: " << timer.get_spent_time()
              << std::endl;

    timer.start_timer();
    LIE tc_scc(grid_size, block_size);
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_3__1_2_3, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tuple_generator_hook reorder_path_host;
    hipMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));
    tc_scc.add_ra(RelationalJoin(edge_2__2_1, FULL, path_3__1_2_3, DELTA,
                                 path_3__1_2_3, reorder_path_host, nullptr,
                                 LEFT, grid_size, block_size, join_detail));
    tc_scc.fixpoint_loop();
    timer.stop_timer();
    // print_tuple_rows(path_3__1_2_3->full, "full path");
    std::cout << "PLEN time: " << timer.get_spent_time() << std::endl;
    std::cout << "join detail: " << std::endl;
    std::cout << "compute size time:  " <<  join_detail[0] <<  std::endl;
    std::cout << "reduce + scan time: " <<  join_detail[1] <<  std::endl;
    std::cout << "fetch result time:  " <<  join_detail[2] <<  std::endl;
    std::cout << "sort time:          " <<  join_detail[3] <<  std::endl;
    std::cout << "build index time:   " <<  join_detail[5] <<  std::endl;
    std::cout << "merge time:         " <<  join_detail[6] <<  std::endl;
    std::cout << "unique time:        " << join_detail[4] + join_detail[7] <<  std::endl;
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

    analysis_bench(argv[1], block_size, grid_size);
    return 0;
}
