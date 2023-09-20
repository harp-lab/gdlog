#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "../include/exception.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

/**
 * @brief binary join, close to local_join in slog's join RA operator
 *
 * @param inner
 * @param outer
 * @param block_size
 */
void binary_join(GHashRelContainer *inner, GHashRelContainer *outer,
                 GHashRelContainer *output_newt, int *reorder_array,
                 int reorder_array_size, JoinDirection direction, int grid_size,
                 int block_size, int iter, float *detail_time) {
    KernelTimer timer;
    // checkCuda(cudaDeviceSynchronize());
    GHashRelContainer *inner_device;
    checkCuda(cudaMalloc((void **)&inner_device, sizeof(GHashRelContainer)));
    cudaMemcpy(inner_device, inner, sizeof(GHashRelContainer),
               cudaMemcpyHostToDevice);
    GHashRelContainer *outer_device;
    checkCuda(cudaMalloc((void **)&outer_device, sizeof(GHashRelContainer)));
    cudaMemcpy(outer_device, outer, sizeof(GHashRelContainer),
               cudaMemcpyHostToDevice);

    u64 *result_counts_array;
    checkCuda(cudaMalloc((void **)&result_counts_array,
                         outer->tuple_counts * sizeof(u64)));

    int *reorder_array_device;
    checkCuda(cudaMalloc((void **)&reorder_array_device,
                         reorder_array_size * sizeof(int)));
    cudaMemcpy(reorder_array_device, reorder_array,
               reorder_array_size * sizeof(int), cudaMemcpyHostToDevice);
    // print_tuple_rows(outer, "outer");

    // std::cout << "inner : " << inner->tuple_counts << " outer: " <<
    // outer->tuple_counts << std::endl; print_hashes(inner, "inner hashes");
    // checkCuda(cudaDeviceSynchronize());
    timer.start_timer();
    get_join_result_size<<<grid_size, block_size>>>(inner_device, outer_device,
                                                    outer->index_column_size,
                                                    result_counts_array, iter);
    checkCuda(cudaDeviceSynchronize());

    u64 total_result_rows =
        thrust::reduce(thrust::device, result_counts_array,
                       result_counts_array + outer->tuple_counts, 0);

    checkCuda(cudaDeviceSynchronize());
    // std::cout << "join result size(non dedup) " << total_result_rows <<
    // std::endl;
    u64 *result_counts_offset;
    checkCuda(cudaMalloc((void **)&result_counts_offset,
                         outer->tuple_counts * sizeof(u64)));
    cudaMemcpy(result_counts_offset, result_counts_array,
               outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToDevice);
    thrust::exclusive_scan(thrust::device, result_counts_offset,
                           result_counts_offset + outer->tuple_counts,
                           result_counts_offset);

    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[0] = timer.get_spent_time();

    timer.start_timer();
    column_type *join_res_raw_data;
    checkCuda(cudaMalloc((void **)&join_res_raw_data, total_result_rows *
                                                          reorder_array_size *
                                                          sizeof(column_type)));
    get_join_result<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size,
        reorder_array_device, reorder_array_size, join_res_raw_data,
        result_counts_array, result_counts_offset, direction);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[1] = timer.get_spent_time();
    cudaFree(result_counts_array);
    cudaFree(result_counts_offset);

    timer.start_timer();
    // // reload newt
    // free_relation(output_newt);
    // newt don't need index
    load_relation(output_newt, reorder_array_size, join_res_raw_data,
                  total_result_rows, 1, 0.6, grid_size, block_size, true, false,
                  false);

    // print_tuple_rows(output_newt, "output_newtr");
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
    // std::cout << "join result size " << output_newt->tuple_counts <<
    // std::endl;

    cudaFree(inner_device);
    cudaFree(outer_device);
}
