#include <chrono>
#include <fstream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.cuh"
#include "../include/relation.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

void transitive_closure(GHashRelContainer *edge_2__2_1,
                        GHashRelContainer *path_2__1_2, int block_size,
                        int grid_size) {
    int output_rel_arity = 2;
    KernelTimer timer;
    // copy struct into gpu
    GHashRelContainer *path_full = path_2__1_2;

    // construct newt/delta for path
    GHashRelContainer *path_newt = new GHashRelContainer();
    column_type foobar_dummy_data[2] = {0, 0};
    load_relation(path_newt, 2, foobar_dummy_data, 0, 1, 0.6, grid_size,
                  block_size);
    GHashRelContainer *path_delta = new GHashRelContainer();
    // before first iteration load all full into delta

    // copy_relation_container(path_delta, path_2__1_2);
    path_delta = path_2__1_2;
    // print_tuple_rows(path_delta,"Delta");
    // print_tuple_rows(edge_2__2_1,"edge_2__2_1");
    int iteration_counter = 0;
    float join_time = 0;
    float merge_time = 0;
    float rebuild_time = 0;
    float flatten_time = 0;
    float set_diff_time = 0;
    float rebuild_delta_time = 0;
    float flatten_full_time = 0;

    float join_get_size_time = 0;
    float join_get_result_time = 0;
    float rebuild_newt_time = 0;
    // the full relation is a buffer of previous delta + full before scc started
    std::vector<GHashRelContainer *> buffered_delta_vectors;
    tuple_type *tuple_full;
    u64 current_full_size = path_full->tuple_counts;
    checkCuda(cudaMalloc((void **)&tuple_full,
                         path_full->tuple_counts * sizeof(tuple_type)));
    cudaMemcpy(tuple_full, path_full->tuples,
               path_full->tuple_counts * sizeof(tuple_type),
               cudaMemcpyDeviceToDevice);

    while (true) {

        // join path delta and edges full
        // TODO: need heuristic for join order
        int reorder_array[2] = {1, 3};
        // print_tuple_rows(path_delta, "Path delta before join");
        timer.start_timer();
        float detail_join_time[3];
        binary_join(edge_2__2_1, path_delta, path_newt, reorder_array, 2,
                    JoinDirection::LEFT, grid_size, block_size,
                    iteration_counter, detail_join_time);
        join_get_size_time += detail_join_time[0];
        join_get_result_time += detail_join_time[1];
        rebuild_newt_time += detail_join_time[2];
        timer.stop_timer();
        join_time += timer.get_spent_time();
        // print_tuple_rows(path_newt, "Path newt after join ");

        // merge delta into full
        timer.start_timer();
        ////
        if (iteration_counter != 0) {
            tuple_type *tuple_full_buf;
            checkCuda(
                cudaMalloc((void **)&tuple_full_buf,
                           (current_full_size + path_delta->tuple_counts) *
                               sizeof(tuple_type)));
            checkCuda(cudaDeviceSynchronize());
            tuple_type *end_tuple_full_buf = thrust::merge(
                thrust::device, tuple_full, tuple_full + current_full_size,
                path_delta->tuples,
                path_delta->tuples + path_delta->tuple_counts, tuple_full_buf,
                tuple_indexed_less(path_delta->index_column_size,
                                   path_delta->arity));
            checkCuda(cudaDeviceSynchronize());
            current_full_size = end_tuple_full_buf - tuple_full_buf;
            cudaFree(tuple_full);
            tuple_full = tuple_full_buf;
        }
        buffered_delta_vectors.push_back(path_delta);
        timer.stop_timer();
        merge_time += timer.get_spent_time();

        // drop the index of delta once merged, because it won't be used in next
        // iter when migrate more general case, this operation need to be put
        // off to end of all RA operation in current iteration
        if (path_delta->index_map != nullptr) {
            cudaFree(path_delta->index_map);
            path_delta->index_map = nullptr;
        }
        if (path_delta->tuples != nullptr) {
            cudaFree(path_delta->tuples);
            path_delta->tuples = nullptr;
        }

        if (path_newt->tuple_counts == 0) {
            // fixpoint
            break;
        }

        // checkCuda(cudaDeviceSynchronize());
        // print_tuple_rows(path_newt, "Path newt before dedup ");
        timer.start_timer();

        tuple_type *deduplicated_newt_tuples;
        checkCuda(cudaMalloc((void **)&deduplicated_newt_tuples,
                             path_newt->tuple_counts * sizeof(tuple_type)));
        //////

        tuple_type *deuplicated_end = thrust::set_difference(
            thrust::device, path_newt->tuples,
            path_newt->tuples + path_newt->tuple_counts, tuple_full,
            tuple_full + current_full_size, deduplicated_newt_tuples,
            tuple_indexed_less(path_full->index_column_size, path_full->arity));
        checkCuda(cudaDeviceSynchronize());
        u64 deduplicate_size = deuplicated_end - deduplicated_newt_tuples;

        if (deduplicate_size == 0) {
            // fixpoint
            break;
        }
        timer.stop_timer();
        set_diff_time += timer.get_spent_time();
        // TODO: optimize here, this can be directly used as tuples in next
        // delta
        column_type *deduplicated_raw;
        checkCuda(cudaMalloc((void **)&deduplicated_raw,
                             deduplicate_size * path_newt->arity *
                                 sizeof(column_type)));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(
            deduplicated_newt_tuples, deduplicated_raw, deduplicate_size,
            path_newt->arity);
        checkCuda(cudaDeviceSynchronize());
        cudaFree(deduplicated_newt_tuples);

        free_relation(path_newt);
        // move newt to delta
        timer.start_timer();
        // deduplicated data is already sorted
        path_delta = new GHashRelContainer();
        load_relation(path_delta, path_full->arity, deduplicated_raw,
                      deduplicate_size, path_full->index_column_size,
                      path_full->index_map_load_factor, grid_size, block_size,
                      true, true, true);
        timer.stop_timer();
        rebuild_delta_time += timer.get_spent_time();

        // print_tuple_rows(path_full, "Path full after load newt");
        // std::cout << "iteration " << iteration_counter << " finish dedup new
        // tuples : " << deduplicate_size
        //           << " newt tuple size: " << path_newt->tuple_counts
        //           << " full counts " <<  current_full_size
        //           << std::endl;
        iteration_counter++;
        // if (iteration_counter == 2) {
        //     // print_tuple_rows(path_newt, "2 t newt");
        //     break;
        // }
    }

    // merge full
    timer.start_timer();
    column_type *new_full_raw_data;
    checkCuda(
        cudaMalloc((void **)&new_full_raw_data,
                   current_full_size * path_full->arity * sizeof(column_type)));
    flatten_tuples_raw_data<<<grid_size, block_size>>>(
        tuple_full, new_full_raw_data, current_full_size, path_full->arity);
    checkCuda(cudaDeviceSynchronize());
    // cudaFree(tuple_merge_buffer);
    load_relation(path_full, path_full->arity, new_full_raw_data,
                  current_full_size, path_full->index_column_size,
                  path_full->index_map_load_factor, grid_size, block_size, true,
                  true);

    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    float merge_full_time = timer.get_spent_time();

    std::cout << "Finished! path has " << path_full->tuple_counts << std::endl;
    std::cout << "Join time: " << join_time << " ; merge time: " << merge_time
              << " ; rebuild full time: " << merge_full_time
              << " ; rebuild delta time: " << rebuild_delta_time
              << " ; set diff time: " << set_diff_time << std::endl;
    std::cout << "Join detail time: " << std::endl;
    std::cout << "get size time: " << join_get_result_time
              << " ; get result time: " << join_get_result_time
              << " ; rebuild newt time: " << rebuild_newt_time << std::endl;
    ;
    // print_tuple_rows(path_full, "Path full at fix point");
    // reach fixpoint
}

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

void graph_bench(const char *dataset_path, int block_size, int grid_size) {
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
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    std::cout << "reversing graph ... " << std::endl;
    for (u64 i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }
    std::cout << "finish reverse graph." << std::endl;

    timer.start_timer();
    GHashRelContainer *edge_2__1_2 = new GHashRelContainer();
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(edge_2__1_2, 2, raw_graph_data, graph_edge_counts, 1, 0.6,
                  grid_size, block_size);
    GHashRelContainer *edge_2__2_1 = new GHashRelContainer();
    load_relation(edge_2__2_1, 2, raw_reverse_graph_data, graph_edge_counts, 1,
                  0.6, grid_size, block_size);
    column_type foobar_dummy_data[2] = {0, 0};
    GHashRelContainer *result_newt = new GHashRelContainer();
    load_relation(result_newt, 2, foobar_dummy_data, 0, 1, 0.6, grid_size,
                  block_size);
    // checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    // double kernel_spent_time = timer.get_spent_time();
    std::cout << "Build hash table time: " << timer.get_spent_time()
              << std::endl;

    // timer.start_timer();
    // // edge_2__2_1 ⋈ path_2__1_2
    // int reorder_array[2] = {1,3};
    // // print_tuple_rows(edge_2__2_1, "edge_2__2_1 before start");
    // binary_join(edge_2__2_1, edge_2__1_2, result_newt, reorder_array, 2,
    // grid_size, block_size, 0); print_tuple_rows(result_newt, "Result newt
    // tuples"); timer.stop_timer(); std::cout << "join time: " <<
    // timer.get_spent_time() << std::endl; std::cout << "Result counts: " <<
    // result_newt->tuple_counts << std::endl;

    timer.start_timer();
    transitive_closure(edge_2__2_1, edge_2__1_2, block_size, grid_size);
    timer.stop_timer();
    std::cout << "TC time: " << timer.get_spent_time() << std::endl;
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

    graph_bench(argv[1], block_size, grid_size);
    return 0;
}