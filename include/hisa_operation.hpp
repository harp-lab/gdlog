
#pragma once

#include "../include/hisa.hpp"

#include <cstdint>
#include <cuco/sentinel.cuh>
#include <rmm/exec_policy.hpp>
#include <stdexcept>
#include <thrust/functional.h>
// #include <cuco/static_map.cuh>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>

#include <cuco/dynamic_map.cuh>

// these are for variadic arity relation
#define DEF_ZIP_ITER_FN(N)                                                     \
    template <typename T> auto zip_iter_##N(DataColumn<T> *col) {              \
        return thrust::make_zip_iterator(                                      \
            iter_tuple_begin(col, std::make_index_sequence<N>{}));             \
    }
DEF_ZIP_ITER_FN(2)
DEF_ZIP_ITER_FN(3)
DEF_ZIP_ITER_FN(4)
DEF_ZIP_ITER_FN(5)

#define DIFFERENCE_ARITY_CASE(N)                                               \
    case N: {                                                                  \
        auto zip_iter = zip_iter_##N(src.data_containers);                     \
        auto zip_iter_target = zip_iter_##N(target.data_containers);           \
        auto zip_iter_result = zip_iter_##N(result.data_containers);           \
        auto res_end = thrust::set_difference(                                 \
            rmm::exec_policy(), zip_iter, zip_iter + src.total_row_size,       \
            zip_iter_target, zip_iter_target + target.total_row_size,          \
            zip_iter_result);                                                  \
        res_size = res_end - zip_iter_result;                                  \
        break;                                                                 \
    }

// set difference
template <typename hisa_t>
void hisa_difference(hisa_t &src, hisa_t &target, hisa_t &result) {
    // check if src and target have the same arity
    if (src.arity != target.arity) {
        throw std::invalid_argument("src and target have different arity");
    }
    size_t arity = src.arity;
    for (size_t i = 0; i < arity; i++) {
        result.data_containers[i].data.resize(target.total_row_size);
    }
    size_t res_size = 0;
    switch (arity) {
        DIFFERENCE_ARITY_CASE(2)
        DIFFERENCE_ARITY_CASE(3)
        DIFFERENCE_ARITY_CASE(4)
        DIFFERENCE_ARITY_CASE(5)
    default:
        throw std::invalid_argument("arity is not supported");
    }
    for (size_t i = 0; i < arity; i++) {
        result.data_containers[i].data.resize(res_size);
        result.data_containers[i].data.shrink_to_fit();
    }
    result.index_container->lex_offset.resize(res_size);
    thrust::sequence(result.index_container->lex_offset.begin(),
                     result.index_container->lex_offset.end());
    result.total_row_size = res_size;
}

template <typename hisa_t>
void hisa_join(hisa_t &inner, hisa_t &outer, hisa_t &result,
               thrust::host_vector<uint32_t> &reorder_mapping) {
    // check if indexed column of inner and outer relation match
    if (inner.index_column_size != outer.index_column_size) {
        throw std::invalid_argument(
            "indexed column of inner and outer relation do not match");
    }

    // create hash table for the inner relation
    // TODO: store the hash table in the index_container
    thrust::device_vector<uint64_t> tmp_inner_hashes(inner.index_container->hashes.size());
    thrust::copy(rmm::exec_policy(), inner.index_container->hashes.begin(),
                 inner.index_container->hashes.end(), tmp_inner_hashes.begin());
    auto tmp_end = thrust::unique(rmm::exec_policy(), tmp_inner_hashes.begin(), tmp_inner_hashes.end());
    tmp_inner_hashes.resize(tmp_end - tmp_inner_hashes.begin());
    auto ziped_inner_hash_iter = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_inner_hashes.begin(), thrust::make_counting_iterator(0)));
    cuco::dynamic_map<uint64_t, uint32_t> inner_hash_table{
        inner.index_container->hashes.size(),
        cuco::empty_key<uint64_t>{std::numeric_limits<uint64_t>::max()},
        cuco::empty_value<uint32_t>{std::numeric_limits<uint32_t>::max()},
    };
    inner_hash_table.insert(ziped_inner_hash_iter,
                            ziped_inner_hash_iter +
                                inner.index_container->hashes.size());
    // free the tmp_inner_hashes
    tmp_inner_hashes.resize(0);
    tmp_inner_hashes.shrink_to_fit();
    thrust::device_vector<uint32_t> matched_hashes_perm(
        outer.index_container->hashes.size());
    assert(matched_hashes.size() == outer.total_row_size);
    inner_hash_table.find(outer.index_container->hashes.begin(),
                          outer.index_container->hashes.end(),
                          matched_hashes_perm.begin());

    // TODO: need better way to compute the join size, this is non coleasced
    thrust::device_vector<uint32_t> join_size(
        outer.index_container->hashes.size());
    thrust::transform(
        rmm::exec_policy(), matched_hashes_perm.begin(),
        matched_hashes_perm.end(), join_size.begin(),
        [inner_indices = inner.index_container->index.data().get(),
         inner_index_size = inner.index_container->index.size(),
         inner_total = inner.total_row_size] __device__(uint32_t x) {
            if (x == std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
            if (x == inner_index_size) {
                return inner_total - inner_indices[x];
            }
            return inner_indices[x + 1] - inner_indices[x];
        });
    thrust::device_vector<uint32_t> matched_hash_indices(outer.total_row_size);
    thrust::gather(rmm::exec_policy(), matched_hashes_perm.begin(),
                   matched_hashes_perm.end(),
                   outer.index_container->index.begin(),
                   matched_hash_indices.begin());
    // reduce the join size
    auto total_join_size =
        thrust::reduce(rmm::exec_policy(), join_size.begin(), join_size.end());
    // prefix sum to get the join result offset for every tuple in d
    result.total_row_size = total_join_size;
    thrust::device_vector<uint32_t> join_offset(join_size.size());
    thrust::exclusive_scan(rmm::exec_policy(), join_size.begin(),
                           join_size.end(), join_offset.begin());
    // write the join result into the result relation
    // the default order is put inner relation columns first, then outer
    for (size_t i = 0; i < reorder_mapping.size(); i++) {
        result.data_containers[i].data.resize(total_join_size);
        auto src_col_i = reorder_mapping[i];
        if (src_col_i < inner.arity) {
            // copy from inner relation
            auto &src_col = inner.data_containers[src_col_i].data;
            // TODO: this is uncoalesced
            thrust::for_each(
                rmm::exec_policy(), thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(outer.total_row_size),
                [src_col = src_col.data().get(),
                 join_size = join_size.data().get(),
                 join_offset = join_offset.data().get(),
                 matched_hash_indices = matched_hash_indices.data().get(),
                 result_col = result.data_containers[i]
                                  .data.data()
                                  .get()] __device__(uint32_t idx) {
                    for (size_t j = 0; j < join_size[idx]; j++) {
                        result_col[join_offset[idx] + j] =
                            src_col[matched_hash_indices[idx] + j];
                    }
                });
        } else {
            // copy from outer relation
            auto &src_col = outer.data_containers[src_col_i - inner.arity].data;
            thrust::for_each(
                rmm::exec_policy(), join_offset.begin(), join_offset.end(),
                [src_col = src_col.data().get(),
                 join_size = join_size.data().get(),
                 join_offset = join_offset.data().get(),
                 matched_hash_indices = matched_hash_indices.data().get(),
                 result_col = result.data_containers[i]
                                  .data.data()
                                  .get()] __device__(uint32_t idx) {
                    for (size_t j = 0; j < join_size[idx]; j++) {
                        result_col[join_offset[idx] + j] = src_col[idx];
                    }
                });
        }
        result.data_containers[i].data.shrink_to_fit();
    }
    // build default lex_offset (unsorted)
    result.index_container->lex_offset.resize(total_join_size);
    thrust::sequence(result.index_container->lex_offset.begin(),
                     result.index_container->lex_offset.end());

    // // finding all matched hash value
    // // the max size of joined hashes are the size of the outer relation's
    // hash values
    // // use result's index_container to store the outer tmp result
    // result.index_container->hashes.resize(outer.index_container->hashes.size());
    // result.index_container->hashes.shrink_to_fit();
    // result.index_container->index.resize(outer.index_container->hashes.size());
    // result.index_container->index.shrink_to_fit();
    // // range query for the outer relation
    // auto end = thrust::set_intersection_by_key(
    //     rmm::exec_policy(),
    //     outer.index_container->hashes.begin(),
    //     outer.index_container->hashes.end(),
    //     inner.index_container->hashes.begin(),
    //     inner.index_container->hashes.end(),
    //     outer.index_container->index.begin(),
    //     result.index_container->hashes.begin(),
    //     result.index_container->index.begin());
    // result.index_container->hashes.resize(end.first -
    // result.index_container->hashes.begin());
    // result.index_container->index.resize(end.second -
    // result.index_container->index.begin()); auto joined_hashes_size =
    // result.index_container->hashes.size(); if (joined_hashes_size == 0) {
    //     result.total_row_size = 0;
    //     return;
    // }
    // // create tmp for inner relation join
    // rmm::device_vector<uint64_t> inner_matched_hashes(joined_hashes_size);
    // rmm::device_vector<uint32_t> inner_matched_perm(joined_hashes_size);
    // // range query for the inner relation
    // end = thrust::set_intersection_by_key(
    //     rmm::exec_policy(),
    //     inner.index_container->hashes.begin(),
    //     inner.index_container->hashes.end(),
    //     result.index_container->hashes.begin(),
    //     result.index_container->hashes.end(),
    //     thrust::make_counting_iterator(0),
    //     inner_matched_hashes.begin(),
    //     inner_matched_perm.begin());
    // inner_matched_hashes.resize(end.first - inner_matched_hashes.begin());
    // inner_matched_perm.resize(end.second - inner_matched_perm.begin());
    // // zip the matched index in inner and outer relation
    // // auto matched_zip_iter = thrust::make_zip_iterator(
    // //     thrust::make_tuple(result.index_container->index.begin(),
    // inner_matched_perm.begin()));
    // // auto matched_zip_iter_end = thrust::make_zip_iterator(
    // //     thrust::make_tuple(result.index_container->index.end(),
    // inner_matched_perm.end()));

    // // check if need join last hash in the inner relation
    // bool do_join_last = inner_matched_perm.size() ==
    // inner.index_container->hashes.size(); rmm::device_vector<uint32_t>
    // inner_matched_index(joined_hashes_size); rmm::device_vector<uint32_t>
    // inner_matched_index_next(joined_hashes_size);
    // thrust::gather(rmm::exec_policy(),
    //                inner_matched_perm.begin(), inner_matched_perm.end(),
    //                inner.index_container->index.begin(),
    //                inner_matched_index.begin());
    // if (!do_join_last) {
    //     thrust::transform(rmm::exec_policy(),
    //                       inner_matched_perm.begin(),
    //                       inner_matched_perm.end(),
    //                       inner_matched_perm.begin(),
    //                       thrust::placeholders::_1 + 1);
    //     thrust::gather(rmm::exec_policy(),
    //                    inner_matched_perm.begin(), inner_matched_perm.end(),
    //                    inner.index_container->index.begin(),
    //                    inner_matched_index_next.begin());
    // } else {
    //     // handle edge case, join on the last hash value
    //     thrust::transform(rmm::exec_policy(),
    //                       inner_matched_perm.begin(),
    //                       inner_matched_perm.end() - 1,
    //                       inner_matched_perm.begin(),
    //                       thrust::placeholders::_1 + 1);
    //     thrust::gather(rmm::exec_policy(),
    //                     inner_matched_perm.begin(), inner_matched_perm.end()
    //                     - 1, inner.index_container->index.begin(),
    //                     inner_matched_index_next.begin());
    //     inner_matched_index_next[joined_hashes_size - 1] =
    //     inner.total_row_size;
    // }

    // thrust::device_vector<uint32_t> result_offset_count(joined_hashes_size);
    // thrust::transform(rmm::exec_policy(),
    //                   inner_matched_index_next.begin(),
    //                   inner_matched_index_next.end(),
    //                   inner_matched_index.begin(),
    //                   result_offset_count.begin(),
    //                   thrust::minus<uint32_t>());
    // inner_matched_index_next.resize(0);
    // inner_matched_index_next.shrink_to_fit();
    // thrust::exclusive_scan(rmm::exec_policy(),
    //                        result_offset_count.begin(),
    //                        result_offset_count.end(),
    //                        result_offset_count.begin());
    // result_offset_count.shrink_to_fit();
}
