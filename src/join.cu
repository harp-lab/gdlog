#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

void RelationalJoin::operator()() {

    bool output_is_tmp = output_rel->tmp_flag;
    GHashRelContainer *inner;
    if (inner_ver == DELTA) {
        inner = inner_rel->delta;
    } else {
        inner = inner_rel->full;
    }
    GHashRelContainer *outer;
    if (outer_ver == DELTA) {
        outer = outer_rel->delta;
    } else if (outer_ver == FULL) {
        outer = outer_rel->full;
    } else {
        // temp relation can be outer relation
        outer = outer_rel->newt;
    }
    int output_arity = output_rel->arity;
    // GHashRelContainer* output = output_rel->newt;

    // std::cout << "inner " << inner_rel->name << " : " << inner->tuple_counts
    //           << " outer " << outer_rel->name << " : " << outer->tuple_counts
    //           << std::endl;
    // print_tuple_rows(inner, "inner");
    // print_tuple_rows(outer, "outer");
    if (outer->tuples == nullptr || outer->tuple_counts == 0) {
        outer->tuple_counts = 0;
        return;
    }
    if (inner->tuples == nullptr || inner->tuple_counts == 0) {
        outer->tuple_counts = 0;
        return;
    }

    KernelTimer timer;
    // checkCuda(cudaDeviceSynchronize());
    GHashRelContainer *inner_device;
    checkCuda(cudaMalloc((void **)&inner_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(inner_device, inner, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));
    GHashRelContainer *outer_device;
    checkCuda(cudaMalloc((void **)&outer_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(outer_device, outer, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));

    tuple_size_t *result_counts_array;
    checkCuda(cudaMalloc((void **)&result_counts_array,
                         outer->tuple_counts * sizeof(tuple_size_t)));
    checkCuda(cudaMemset(result_counts_array, 0,
                         outer->tuple_counts * sizeof(tuple_size_t)));

    // print_tuple_rows(outer, "inber");
    // checkCuda(cudaDeviceSynchronize());
    timer.start_timer();
    checkCuda(cudaDeviceSynchronize());
    get_join_result_size<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size, tuple_generator,
        tuple_pred, result_counts_array);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    this->detail_time[0] += timer.get_spent_time();

    timer.start_timer();
    tuple_size_t total_result_rows = 0;
    for (tuple_size_t i = 0; i < outer->tuple_counts; i = i + MAX_REDUCE_SIZE) {
        tuple_size_t reduce_size = MAX_REDUCE_SIZE;
        if (i + MAX_REDUCE_SIZE > outer->tuple_counts) {
            reduce_size = outer->tuple_counts - i;
        }
        tuple_size_t reduce_v = thrust::reduce(
            thrust::device, result_counts_array + i,
            result_counts_array + i + reduce_size, 0);
        total_result_rows += reduce_v;
        // checkCuda(cudaDeviceSynchronize());
    }
    
    std::cout << output_rel->name << "   " << outer->index_column_size
              << " join result size(non dedup) " << total_result_rows
              << std::endl;
    // print_memory_usage();
    tuple_size_t *result_counts_offset;
    checkCuda(cudaMalloc((void **)&result_counts_offset,
                         outer->tuple_counts * sizeof(tuple_size_t)));
    checkCuda(cudaMemcpy(result_counts_offset, result_counts_array,
                         outer->tuple_counts * sizeof(tuple_size_t),
                         cudaMemcpyDeviceToDevice));
    thrust::exclusive_scan(thrust::device, result_counts_offset,
                           result_counts_offset + outer->tuple_counts,
                           result_counts_offset);

    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[1] += timer.get_spent_time();

    timer.start_timer();
    column_type *join_res_raw_data;
    u64 join_res_raw_data_mem_size =
        total_result_rows * output_arity * sizeof(column_type);
    checkCuda(
        cudaMalloc((void **)&join_res_raw_data, join_res_raw_data_mem_size));
    checkCuda(cudaMemset(join_res_raw_data, 0, join_res_raw_data_mem_size));
    get_join_result<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size, tuple_generator,
        tuple_pred, output_arity, join_res_raw_data, result_counts_array,
        result_counts_offset, direction);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[2] += timer.get_spent_time();
    checkCuda(cudaFree(result_counts_array));
    checkCuda(cudaFree(result_counts_offset));

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};
    // // reload newt
    // free_relation(output_newt);
    // newt don't need index
    if (output_rel->newt->tuples == nullptr ||
        output_rel->newt->tuple_counts == 0) {
        if (disable_load) {
            return;
        }
        if (!output_is_tmp) {
            load_relation_container(
                output_rel->newt, output_arity, join_res_raw_data,
                total_result_rows, output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, false, false);
        } else {
            // temporary relation doesn't need index nor sort
            // std::cout << "use tmp >>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<
            // std::endl;
            load_relation_container(
                output_rel->newt, output_arity, join_res_raw_data,
                total_result_rows, output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, true, false);
            output_rel->newt->tmp_flag = true;
        }
        checkCuda(cudaDeviceSynchronize());
        detail_time[3] += load_relation_container_time[0];
        detail_time[4] += load_relation_container_time[1];
        detail_time[5] += load_relation_container_time[2];
        // print_tuple_rows(output_rel->newt, "newt after join");
    } else {
        // TODO: handle the case out put relation is temp relation
        // data in current newt, merge
        if (!output_is_tmp) {
            GHashRelContainer *newt_tmp = new GHashRelContainer(
                output_rel->arity, output_rel->index_column_size,
                output_rel->dependent_column_size);
            GHashRelContainer *old_newt = output_rel->newt;
            load_relation_container(
                newt_tmp, output_arity, join_res_raw_data, total_result_rows,
                output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, false, false);
            detail_time[3] += load_relation_container_time[0];
            detail_time[4] += load_relation_container_time[1];
            detail_time[5] += load_relation_container_time[2];
            // checkCuda(cudaDeviceSynchronize());
            tuple_type *tp_buffer;
            u64 tp_buffer_mem_size =
                (newt_tmp->tuple_counts + old_newt->tuple_counts) *
                sizeof(tuple_type);
            checkCuda(cudaMalloc((void **)&tp_buffer, tp_buffer_mem_size));
            cudaMemset(tp_buffer, 0, tp_buffer_mem_size);
            timer.start_timer();
            tuple_type *tp_buffer_end = thrust::merge(
                thrust::device, newt_tmp->tuples,
                newt_tmp->tuples + newt_tmp->tuple_counts, old_newt->tuples,
                old_newt->tuples + old_newt->tuple_counts, tp_buffer,
                tuple_indexed_less(output_rel->index_column_size,
                                   output_rel->arity));
            // checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            detail_time[6] += timer.get_spent_time();
            // cudaFree(newt_tmp->tuples);
            // cudaFree(old_newt->tuples);
            timer.start_timer();
            tp_buffer_end =
                thrust::unique(thrust::device, tp_buffer, tp_buffer_end,
                               t_equal(output_rel->arity));
            checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            detail_time[7] += timer.get_spent_time();
            tuple_size_t new_newt_counts = tp_buffer_end - tp_buffer;
            // std::cout << " >>>>>>>>>> " << new_newt_counts *
            // output_rel->arity * sizeof(column_type) << std::endl;

            timer.start_timer();
            column_type *new_newt_raw;
            u64 new_newt_raw_mem_size =
                new_newt_counts * output_rel->arity * sizeof(column_type);
            checkCuda(
                cudaMalloc((void **)&new_newt_raw, new_newt_raw_mem_size));
            checkCuda(cudaMemset(new_newt_raw, 0, new_newt_raw_mem_size));
            flatten_tuples_raw_data<<<grid_size, block_size>>>(
                tp_buffer, new_newt_raw, new_newt_counts, output_rel->arity);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            detail_time[4] += timer.get_spent_time();
            checkCuda(cudaFree(tp_buffer));
            free_relation_container(old_newt);
            free_relation_container(newt_tmp);
            // TODO: free newt_tmp pointer
            load_relation_container(
                output_rel->newt, output_arity, new_newt_raw, new_newt_counts,
                output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, true, false);
            checkCuda(cudaDeviceSynchronize());
        } else {
            // output relation is tmp relation, directly merge without sort
            GHashRelContainer *old_newt = output_rel->newt;
            column_type *newt_tmp_raw;
            u64 newt_tmp_raw_mem_size =
                (old_newt->tuple_counts + total_result_rows) *
                output_rel->arity * sizeof(column_type);
            tuple_size_t new_newt_counts =
                old_newt->tuple_counts + total_result_rows;
            checkCuda(
                cudaMalloc((void **)&newt_tmp_raw, newt_tmp_raw_mem_size));
            checkCuda(cudaMemcpy(newt_tmp_raw, old_newt->data_raw,
                                 old_newt->tuple_counts * old_newt->arity *
                                     sizeof(column_type),
                                 cudaMemcpyDeviceToDevice));
            checkCuda(cudaMemcpy(
                &(newt_tmp_raw[old_newt->tuple_counts * old_newt->arity]),
                join_res_raw_data,
                total_result_rows * output_rel->arity * sizeof(column_type),
                cudaMemcpyDeviceToDevice));
            free_relation_container(old_newt);
            checkCuda(cudaFree(join_res_raw_data));
            load_relation_container(
                output_rel->newt, output_arity, newt_tmp_raw, new_newt_counts,
                output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, true, false);
            checkCuda(cudaDeviceSynchronize())
        }

        detail_time[3] += load_relation_container_time[0];
        detail_time[4] += load_relation_container_time[1];
        detail_time[5] += load_relation_container_time[2];
        // print_tuple_rows(output_rel->newt, "join merge newt");
        // delete newt_tmp;
    }

    // print_tuple_rows(output_rel->newt, "output_newtr");
    // checkCuda(cudaDeviceSynchronize());
    // std::cout << output_rel->name << " join result size " <<
    // output_rel->newt->tuple_counts <<std::endl;

    checkCuda(cudaFree(inner_device));
    checkCuda(cudaFree(outer_device));
}
