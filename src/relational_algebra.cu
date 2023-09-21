#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "../include/exception.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"
#include "../include/print.cuh"

void RelationalJoin::operator()() {

    GHashRelContainer* inner;
    if (inner_ver == DELTA) {
        inner = inner_rel->delta;
    } else {
        inner = inner_rel->full;
    }
    GHashRelContainer* outer;
    if (outer_ver == DELTA) {
        outer = outer_rel->delta;
    } else {
        outer = outer_rel->full;
    }
    int output_arity = output_rel->arity;
    // GHashRelContainer* output = output_rel->newt;

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

    // std::cout << "inner : " << inner->tuple_counts << " outer: " << outer->tuple_counts << std::endl;
    checkCuda(cudaDeviceSynchronize());
    timer.start_timer();
    get_join_result_size<<<grid_size, block_size>>>(inner_device, outer_device,
                                                    outer->index_column_size,
                                                    result_counts_array);
    checkCuda(cudaDeviceSynchronize());

    u64 total_result_rows =
        thrust::reduce(thrust::device, result_counts_array,
                       result_counts_array + outer->tuple_counts, 0);

    checkCuda(cudaDeviceSynchronize());
    // std::cout << outer_rel->name << output_arity << " join result size(non dedup) " << total_result_rows << std::endl;
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
    // detail_time[0] = timer.get_spent_time();

    timer.start_timer();
    column_type *join_res_raw_data;
    checkCuda(
        cudaMalloc((void **)&join_res_raw_data,
                   total_result_rows * output_arity * sizeof(column_type)));
    get_join_result<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size, tuple_generator,
        output_arity, join_res_raw_data, result_counts_array,
        result_counts_offset, direction);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    // detail_time[1] = timer.get_spent_time();
    cudaFree(result_counts_array);
    cudaFree(result_counts_offset);

    timer.start_timer();
    // // reload newt
    // free_relation(output_newt);
    // newt don't need index
    load_relation_container(output_rel->newt, output_arity, join_res_raw_data,
                            total_result_rows, output_rel->index_column_size, 0.8, grid_size, block_size,
                            true, false, false);

    // print_tuple_rows(output_rel->newt, "output_newtr");
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    // detail_time[2] = timer.get_spent_time();
    // std::cout << output_rel->name << " join result size " << output_rel->newt->tuple_counts <<std::endl;

    cudaFree(inner_device);
    cudaFree(outer_device);
}
