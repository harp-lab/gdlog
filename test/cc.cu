#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

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

//////////////////////////////////////////////////////////////////

__device__ void reorder_path(tuple_type inner, tuple_type outer,
                             tuple_type newt) {
    newt[0] = inner[1];
    newt[1] = outer[1];
};
__device__ tuple_generator_hook reorder_path_device = reorder_path;

__device__ void cp_00(tuple_type src, tuple_type dest) {
    dest[0] = src[0];
    dest[1] = src[0];
}
__device__ tuple_copy_hook cp_00_device = cp_00;

__device__ bool compare_dep_hook_1(tuple_type t1, tuple_type t2) {
    return t1[0] < t2[0];
}
__device__ dependency_order compare_dep_hook_1_device = compare_dep_hook_1;

void analysis_bench(const char *dataset_path, int block_size, int grid_size) {
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

    timer.start_timer();
    Relation *edge_2__1_2 = new Relation();
    // cudaMallocHost((void **)&edge_2__2_1, sizeof(Relation));
    Relation *cc_2__1_2 = new Relation();
    cc_2__1_2->index_flag = false;
    // cudaMallocHost((void **)&path_2__1_2, sizeof(Relation));
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(cc_2__1_2, "cc_2__1_2", 2, nullptr, 0, 1, 0, grid_size,
                  block_size);
    load_relation(edge_2__1_2, "edge_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    timer.stop_timer();
    // double kernel_spent_time = timer.get_spent_time();
    std::cout << "Build hash table time: " << timer.get_spent_time()
              << std::endl;

    LIE init_scc(grid_size, block_size);
    init_scc.add_relations(edge_2__1_2, true);
    init_scc.add_relations(cc_2__1_2, false);
    tuple_copy_hook cp_00_host;
    cudaMemcpyFromSymbol(&cp_00_host, cp_00_device, sizeof(tuple_copy_hook));
    init_scc.add_ra(RelationalCopy(edge_2__1_2, FULL, cc_2__1_2, cp_00_host,
                                   nullptr, grid_size, block_size));
    init_scc.fixpoint_loop();

    timer.start_timer();
    LIE tc_scc(grid_size, block_size);
    tc_scc.add_relations(edge_2__1_2, true);
    tc_scc.add_relations(cc_2__1_2, false);
    dependency_order compare_dep_hook_1_host;
    cudaMemcpyFromSymbol(&compare_dep_hook_1_host, compare_dep_hook_1_device,
                         sizeof(dependency_order));
    cc_2__1_2->dep_pred = compare_dep_hook_1_host;
    float join_time[3];
    tuple_generator_hook reorder_path_host;
    cudaMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));

    tc_scc.add_ra(RelationalJoin(edge_2__1_2, FULL, cc_2__1_2, DELTA, cc_2__1_2,
                                 reorder_path_host, nullptr, LEFT, grid_size,
                                 block_size, join_time));

    tc_scc.fixpoint_loop();

    timer.stop_timer();
    std::cout << "Path counts " << cc_2__1_2->full->tuple_counts << std::endl;
    // print_tuple_rows(path_2__2_1->full, "full");
    std::cout << "CC time: " << timer.get_spent_time() << std::endl;
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
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