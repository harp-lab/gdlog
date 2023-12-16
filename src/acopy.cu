#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

struct acopy_init_unsort_func {
    column_type* data_raw;
    int output_arity;

    acopy_init_unsort_func(column_type* data_raw, int output_arity)
        : data_raw(data_raw), output_arity(output_arity) {}
    __device__ tuple_type operator()(int index) {
        return data_raw + index * output_arity;
    }
};

void RelationalACopy::operator()() {

    GHashRelContainer *src = src_rel->newt;
    GHashRelContainer *dest = dest_rel->newt;
    std::cout << "ACopy " << src_rel->name << " to " << dest_rel->name
              << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    int output_arity = dest_rel->arity;

    dest->tuples.resize(src->tuple_counts);
    dest->tuple_counts = src->tuple_counts;
    dest->data_raw.resize(src->data_raw_row_size);
    dest->data_raw_row_size = src->data_raw_row_size;
    get_copy_result<<<grid_size, block_size>>>(
        src->tuples.data().get(), dest->data_raw.data().get(), output_arity, src->tuple_counts,
        tuple_generator);
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
    // init tuples with (k*output_arity + dest->data_raw.data()) where k in range (0 ~ src->tuple_counts) using thrust
    thrust::counting_iterator<tuple_size_t> index_sequence_begin(0);
    thrust::counting_iterator<tuple_size_t> index_sequence_end(src->tuple_counts);
    thrust::transform(thrust::device, index_sequence_begin, index_sequence_end,
                      dest->tuples.begin(),
                      acopy_init_unsort_func(dest->data_raw.data().get(), output_arity));
}
