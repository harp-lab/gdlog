#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "../include/exception.h"
#include "../include/print.h"
#include "../include/relational_algebra.h"
#include "../include/timer.h"

void RelationalACopy::operator()() {

    GHashRelContainer *src = src_rel->newt;
    GHashRelContainer *dest = dest_rel->newt;
    std::cout << "ACopy " << src_rel->name << " to " << dest_rel->name
              << std::endl;

    if (src->tuple_counts == 0) {
        free_relation_container(dest);
        dest->tuple_counts = 0;
        return;
    }

    int output_arity = dest_rel->arity;
    column_type *copied_raw_data;
    u64 copied_raw_data_size =
        src->tuple_counts * output_arity * sizeof(column_type);
    checkHip(hipMalloc((void **)&copied_raw_data, copied_raw_data_size));
    checkHip(hipMemset(copied_raw_data, 0, copied_raw_data_size));
    get_copy_result<<<grid_size, block_size>>>(src->tuples, copied_raw_data,
                                               output_arity, src->tuple_counts,
                                               tuple_generator);
    checkHip(hipGetLastError());
    checkHip(hipDeviceSynchronize());

    free_relation_container(dest);
    float detail_time[5] = {0, 0, 0, 0, 0};
    // TODO: swap to repartition_relation_index in future
    load_relation_container(dest, dest->arity, copied_raw_data,
                            src->tuple_counts, src->index_column_size,
                            dest->dependent_column_size, 0.8, grid_size,
                            block_size, detail_time, true, false, true);
    checkHip(hipDeviceSynchronize());
    // print_tuple_rows(dest, "delta");
    // merge delta to full immediately here
    // dest_rel->flush_delta(grid_size, block_size);
    // std::cout << dest->tuple_counts << std::endl;
    // print_tuple_rows(dest, "acopied");
}
