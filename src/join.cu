#include <iostream>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relational_algebra.cuh"
#include "../include/timer.cuh"

void RelationalJoin::operator()() {
    compute_join(result_counts_buf_default, result_offset_buf_default);
}

void RelationalJoin::operator()(counting_buf_t &result_counts_buf,
                                counting_buf_t &result_offset_buf) {
    compute_join(result_counts_buf, result_offset_buf);
}
void RelationalJoin::compute_join(counting_buf_t &result_counts_buf,
                                  counting_buf_t &result_offset_buf) {
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
    if (outer->tuple_counts == 0) {
        return;
    }
    if (inner->tuple_counts == 0) {
        return;
    }

    KernelTimer timer;
    // checkCuda(cudaStreamSynchronize(0));
    timer.start_timer();
    result_counts_buf.resize(outer->tuple_counts);

    // print_tuple_rows(outer, "inber");
    // checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaStreamSynchronize(0));
    get_join_result_size<<<grid_size, block_size>>>(
        inner->index_map.data().get(), inner->index_map_size,
        inner->tuple_counts, inner->tuples.data().get(),
        outer->tuples.data().get(), outer->tuple_counts,
        outer->index_column_size, outer->index_column_size, tuple_generator,
        tuple_pred, result_counts_buf.data().get());
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(0));
    timer.stop_timer();
    this->detail_time[0] += timer.get_spent_time();

    // thrust::for_each(thrust::device, result_counts_buf.begin(),
    //                  result_counts_buf.end(), [] __device__(tuple_size_t & x)
    //                  {
    //                      printf("result_counts_buf %d\n", x);
    //                  });

    timer.start_timer();
    tuple_size_t total_result_rows = 0;
    // NOTE: this is for bug fix in thrust, if still has bug, use the following
    // code
    for (tuple_size_t i = 0; i < outer->tuple_counts; i = i + MAX_REDUCE_SIZE) {
        tuple_size_t reduce_size = MAX_REDUCE_SIZE;
        if (i + MAX_REDUCE_SIZE > outer->tuple_counts) {
            reduce_size = outer->tuple_counts - i;
        }
        tuple_size_t reduce_v =
            thrust::reduce(thrust::device, result_counts_buf.begin() + i,
                           result_counts_buf.begin() + i + reduce_size, 0);
        total_result_rows += reduce_v;
    }
    // total_result_rows = thrust::reduce(thrust::device,
    // result_counts_buf.begin(),
    //                                    result_counts_buf.end(), 0);

    cudaDeviceSynchronize();
    std::cout << output_rel->name << "   " << outer->index_column_size
              << " join result size(non dedup) " << total_result_rows
              << std::endl;
    // print_memory_usage();
    result_offset_buf.resize(outer->tuple_counts);
    thrust::copy(thrust::device, result_counts_buf.begin(),
                 result_counts_buf.end(), result_offset_buf.begin());
    thrust::exclusive_scan(thrust::device, result_offset_buf.begin(),
                           result_offset_buf.end(), result_offset_buf.begin());
    timer.stop_timer();
    detail_time[1] += timer.get_spent_time();
    // thrust::for_each(thrust::device, result_counts_buf.begin(),
    //                  result_counts_buf.end(), [] __device__(tuple_size_t & x)
    //                  {
    //                      printf("result_counts_buf %d\n", x);
    //                  });
    // print result_offset_buf
    // thrust::for_each(thrust::device, result_offset_buf.begin(),
    //                  result_offset_buf.end(), [] __device__(tuple_size_t & x)
    //                  {
    //                      printf("result_offset_buf %d\n", x);
    //                  });
    // print result_counts_buf

    timer.start_timer();
    output_rel->newt->tuple_counts = total_result_rows;
    output_rel->newt->data_raw.resize(total_result_rows * output_arity);
    output_rel->newt->data_raw_row_size = total_result_rows;
    get_join_result<<<grid_size, block_size>>>(
        inner->index_map.data().get(), inner->index_map_size,
        inner->tuple_counts, inner->tuples.data().get(),
        outer->tuples.data().get(), outer->tuple_counts,
        outer->index_column_size, outer->index_column_size, tuple_generator,
        tuple_pred, output_arity, output_rel->newt->data_raw.data().get(),
        result_counts_buf.data().get(), result_offset_buf.data().get(),
        direction);
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
}
