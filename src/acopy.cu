#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

void RelationalACopy::operator()() {
    
    GHashRelContainer *src = src_rel->newt;
    GHashRelContainer *dest = dest_rel->newt;
    std::cout << "ACopy " << src_rel->name << " to " << dest_rel->name << std::endl;

    if (src->tuple_counts == 0) {
        free_relation_container(dest);
        dest->tuple_counts = 0;
        return;
    }

    int output_arity = dest_rel->arity;
    column_type *copied_raw_data;
    checkCuda(
        cudaMalloc((void **)&copied_raw_data,
                   src->tuple_counts * output_arity * sizeof(column_type)));
    get_copy_result<<<grid_size, block_size>>>(src->tuples, copied_raw_data,
                                               output_arity, src->tuple_counts,
                                               tuple_generator);
    checkCuda(cudaDeviceSynchronize());
    
    free_relation_container(dest);
    load_relation_container(dest, dest->arity, copied_raw_data,
                            src->tuple_counts, src->index_column_size,
                            dest->dependent_column_size, 0.8, grid_size,
                            block_size, true, false, true);
    checkCuda(cudaDeviceSynchronize());
    // print_tuple_rows(dest, "delta");
    // merge delta to full immediately here
    // dest_rel->flush_delta(grid_size, block_size);
    // std::cout << dest->tuple_counts << std::endl;
    // print_tuple_rows(dest, "acopied");
}
