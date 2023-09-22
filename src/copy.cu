#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

void RelationalCopy::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else {
        src = src_rel->full;
    }
    GHashRelContainer *dest = dest_rel->newt;

    int output_arity = dest_rel->arity;
    column_type *copied_raw_data;
    checkCuda(
        cudaMalloc((void **)&copied_raw_data,
                   src->tuple_counts * output_arity * sizeof(column_type)));
    get_copy_result<<<grid_size, block_size>>>(src->tuples, copied_raw_data,
                                               output_arity, src->tuple_counts,
                                               tuple_generator);

    if (dest->tuples == nullptr && dest->tuple_counts == 0) {
        free_relation_container(dest);
        load_relation_container(dest, dest->arity, copied_raw_data,
                                src->tuple_counts, src->index_column_size, 0.8,
                                grid_size, block_size, true, false, false);
    } else {
        GHashRelContainer *tmp =
            new GHashRelContainer(dest->arity, dest->index_column_size);
        load_relation_container(tmp, dest->arity, copied_raw_data,
                                src->tuple_counts, src->index_column_size, 0.8,
                                grid_size, block_size, true, false, false);
        // merge to newt
        GHashRelContainer *old_newt = dest;
        tuple_type *tp_buffer;
        checkCuda(cudaMalloc((void **)&tp_buffer,
                             (old_newt->tuple_counts + src->tuple_counts) *
                                 sizeof(tuple_type)));
        tuple_type *tp_buffer_end = thrust::merge(
            thrust::device, old_newt->tuples,
            old_newt->tuples + old_newt->tuple_counts, tmp->tuples,
            tmp->tuples + tmp->tuple_counts, tp_buffer,
            tuple_indexed_less(dest->index_column_size, output_arity));
        checkCuda(cudaDeviceSynchronize());
        cudaFree(tmp->tuples);
        cudaFree(old_newt->tuples);
        tp_buffer_end = thrust::unique(thrust::device, tp_buffer, tp_buffer_end,
                                       t_equal(output_arity));
        checkCuda(cudaDeviceSynchronize());
        u64 new_newt_counts = tp_buffer_end - tp_buffer;
        column_type *new_newt_raw;
        checkCuda(cudaMalloc((void **)&tp_buffer,
                             (tmp->tuple_counts + old_newt->tuple_counts) *
                                 output_arity * sizeof(column_type)));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(
            tp_buffer, new_newt_raw, new_newt_counts, output_arity);
        checkCuda(cudaDeviceSynchronize());
        cudaFree(tp_buffer);
        free_relation_container(old_newt);
        free_relation_container(tmp);
        load_relation_container(dest, output_arity, new_newt_raw,
                                new_newt_counts, dest->index_column_size,
                                0.8, grid_size, block_size);
        // delete tmp;
    }
}
