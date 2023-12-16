#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
// #include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"
#include "../include/relation.cuh"

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

    dest->tuple_counts = src->tuple_counts;
    dest->data_raw.resize(src->data_raw.size());
    dest->data_raw_row_size = src->tuple_counts;
    get_copy_result<<<grid_size, block_size>>>(
        src->tuples.data().get(), dest->data_raw.data().get(), output_arity,
        src->tuple_counts, tuple_generator);
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
