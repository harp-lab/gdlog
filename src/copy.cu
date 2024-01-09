#include <iostream>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
// #include <thrust/transform.h>
#include <rmm/exec_policy.hpp>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relation.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

// struct CopyTupleGenerator {

//     __device__ void operator()(tuple_type src, tuple_type dest) {
//         for (int i = 0; i < arity; i++) {
//             dest[i] = src[i];
//         }
//     }
// };

void RelationalCopy::operator()() {
    checkCuda(cudaStreamSynchronize(0));
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else {
        src = src_rel->full;
    }
    GHashRelContainer *dest = dest_rel->newt;
    std::cout << "Copy " << src_rel->name << " to " << dest_rel->name
              << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    int output_arity = dest_rel->arity;

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};

    tuple_size_t existing_newt_size = dest->tuple_counts;
    dest->tuple_counts = existing_newt_size + src->tuple_counts;
    dest->data_raw.resize(dest->tuple_counts * output_arity);
    dest->data_raw_row_size = existing_newt_size + src->tuple_counts;
    // get_copy_result<<<grid_size, block_size>>>(
    //     src->tuples.data().get(), dest->data_raw.data().get(), output_arity,
    //     src->tuple_counts, tuple_generator);
    thrust::copy(rmm::exec_policy(), src->data_raw.begin(), src->data_raw.end(),
                 dest->data_raw.begin() + existing_newt_size * output_arity);
    thrust::counting_iterator<tuple_size_t> index_sequence_begin(0);
    thrust::counting_iterator<tuple_size_t> index_sequence_end(
        src->tuple_counts);
    thrust::for_each(
        rmm::exec_policy(), index_sequence_begin, index_sequence_end,
        [tp_gen = tuple_generator, arity = output_arity,
         dest_raw =
             dest->data_raw.data().get() + existing_newt_size * output_arity,
         src_raw = src->data_raw.data().get()] __device__(tuple_size_t index) {
            (*tp_gen)(src_raw + index * arity, dest_raw + index * arity);
        });

    // TODO: rewrite this using thrust
    // thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
    //     thrust::make_counting_iterator(src->tuple_counts),
    //     [=] __device__ (int i) {
    //         tuple_type dest_tuple = dest->data_raw + (i * output_arity);
    //         (*tuple_generator)(, dest_tuple);
    //     });

    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
}
