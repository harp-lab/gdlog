/**
 * A GPU prefix tree implementation.
 */

#pragma once
#include <cstdint>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <timer.cuh>

using column_data_type_default = int;
using column_container_type_default =
    rmm::device_vector<column_data_type_default>;
using column_index_type_default =
    rmm::device_vector<column_container_type_default>;

template <typename container_type, typename column_index_type>
struct IndexColumn {
    // uint64_t size;
    // all value in current column, this vector is unique
    container_type data;
    // need a better hash map?
    rmm::device_vector<uint32_t> hashes;
    rmm::device_vector<uint32_t> index;
    // tuple permutation in hash order
    rmm::device_vector<uint32_t> offsets;
    // tuple permutation in lexical order
    rmm::device_vector<uint32_t> lex_offset;
};

template <typename container_type> struct DataColumn {
    container_type data;
};

template <typename col_type = column_data_type_default,
          typename container_type = rmm::device_vector<col_type>,
          typename column_index_type = rmm::device_vector<col_type>>
struct HISA {
    using trie_column_type = IndexColumn<container_type, column_index_type>;
    trie_column_type *index_container;
    using data_column_type = DataColumn<container_type>;
    data_column_type *data_containers;

    using self_type = HISA<col_type, container_type, column_index_type>;

    int arity;
    int index_column_size;
    int data_column_size;
    col_type total_row_size;

    bool compress_flag = true;
    bool indexed_flag = true;

    // this flag indicate if row order in each data columns
    // follow the index ording
    bool sort_data_column_flag = true;

    HISA(int arity, int index_column_size, bool compress_flag = true)
        : arity(arity), index_column_size(index_column_size),
          compress_flag(compress_flag) {
        index_container = new IndexColumn<container_type, column_index_type>;
        data_containers = new DataColumn<container_type>[arity];
        data_column_size = arity;
    }

    ~HISA() {
        delete index_container;
        delete[] data_containers;
    }

    float detail_time[10] = {0.0f};
    // 0 for sorting time
    // 1 for copy time
    // 2 for unique time
    // 3 for index time

    // build trie from data, data is a vertical table
    // template <typename raw_data_type>
    void build(rmm::device_vector<col_type> &flatten_data, uint64_t row_size) {
        KernelTimer timer;
        // sort flatten_data first
        timer.start();
        rmm::device_vector<col_type> sorted_raw_row_index(row_size);
        sort_raw_data(flatten_data, sorted_raw_row_index, row_size, arity);
        timer.stop();
        detail_time[0] += timer.get_spent_time();

        timer.start();
        _load(flatten_data, row_size, sorted_raw_row_index);
        timer.stop();
        detail_time[1] += timer.get_spent_time();

        timer.start();
        _dedup();
        timer.stop();
        detail_time[2] += timer.get_spent_time();

        timer.start();
        if (indexed_flag) {
            timer.start();
            build_index();
            timer.stop();
        }
        timer.start();
        detail_time[3] += timer.get_spent_time();
    }

    // merge and deduplicates, leave set_differece tuples in target
    // need rebuild index for target after use this function
    void difference_and_merge(self_type &target) {
        rmm::device_vector<col_type> merged_index(total_row_size +
                                                  target.total_row_size);
        merged_index.shrink_to_fit();
        _merge_permutation(target, merged_index);
        // _reorder_merge(target, merged_index);
        _unordered_merge(target, merged_index);

        rmm::device_vector<bool> unique_dedup(row_size);
        thrust::fill(rmm::exec_policy(), unique_dedup.begin(),
                     unique_dedup.end(), false);
        _find_dup(unique_dedup);
        // drop all dups in permutation
        auto new_end = thrust::remove_if(
            rmm::exec_policy(), merged_index.begin(), merged_index.end(),
            unique_dedup.begin(),
            [] __device__(bool is_unique) { return !is_unique; });
        auto new_perm_size = new_end - merged_index.begin();
        merged_index.resize(new_perm_size);
        _remove_tuples_in_vec(unique_dedup);
        // partition to get all perm value originally in target

        // TODO: populate existing ID using unique_dedup

        rmm::device_vector<col_type> row_index_perm(total_row_size);
        thrust::sequence(row_index_perm.begin(), row_index_perm.end());
        auto new_end = thrust::partition(
            rmm::exec_policy(), row_index_perm.begin(), row_index_perm.end(),
            [total_row_size, idx_ptr = merged_index.begin()] __device__(int p) {
                return idx_ptr[p] >= total_row_size;
            });
        auto new_size_delta = new_end - row_index_perm.begin();
        row_index_perm.resize(new_size_delta);
        row_index_perm.shrink_to_fit();

        for (int i = arity - 1; i >= 0; i++) {
            // need faster copy
            auto &cur_column = data_containers + i;
            auto &target_column = target.data_containers + i;
            target_column->data.resize(row_index_perm.size());
            thrust::gather(rmm::exec_policy(), row_index_perm.start(),
                           row_index_perm.end(), cur_column->data.begin(),
                           target_column->data.begin());
        }
    }

    // merge and store merged permutation in merged_index
    // in permutation, 0 ~ total_row_size comes from `this`,
    // total_row_size ~ total_row_size + target->total_row_size comes from
    // target
    void _merge_permutation(self_type &target,
                            rmm::device_vector<col_type> &merged_index) {
        rmm::device_vector<col_type> tmp_this_col(total_row_size);
        rmm::device_vector<col_type> tmp_target_col(target.total_row_size);

        rmm::device_vector<col_type> tmp_this_index(total_row_size);
        rmm::device_vector<col_type> tmp_target_index(target.total_row_size);
        // init idxs

        thrust::copy(rmm::exec_policy(), index_container->lex_offset.begin(),
                     index_container->lex_offset.end(), tmp_this_index.begin());
        thrust::copy(
            rmm::exec_policy(), target.index_container->lex_offset.begin(),
            target.index_container->lex_offset.end(), tmp_target_index.begin());

        // a bit vector halding useless merged tmp value, this just to conform
        // thrust
        rmm::device_vector<bool> useless(total_row_size +
                                         target.total_row_size);

        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            data_column_type *target_column =
                target.data_containers + cur_column_idx;
            // copy data to tmp
            thurst::gather(rmm::exec_policy(), tmp_this_index.begin(),
                           tmp_this_index.end(), cur_column->data.begin(),
                           tmp_this_col.begin());
            // do we have a better way to gather this from permutation
            thrust::transform(
                rmm::exec_policy(), target_column->data.begin(),
                target_column->data.end(), tmp_target_col.begin(),
                [raw_data_ptr = target_column->data.begin(),
                 this_size =
                     cur_column->data.size()] __device__(col_type data) {
                    return raw_data_ptr[data - this_size];
                });
            tmp_target_col.shrink_to_fit();
            tmp_this_col.shrink_to_fit();

            // merge data column
            // NOTE: target need to be the first and this need to be the second
            // so that in dedup, we always leave `this` tuples
            thrust::merge_by_key(rmm::exec_policy(), tmp_target_col.begin(),
                                 tmp_target_col.end(), tmp_this_col.begin(),
                                 tmp_this_col.end(), tmp_target_index.begin(),
                                 tmp_this_index.begin(), useless.begin(),
                                 merged_index.begin());
            merged_index.shrink_to_fit();
        }
    }

    // this merge will not change any data, but just merge them directly
    // put target at the end
    void _unordered_merge(self_type &target,
                          rmm::device_vector<col_type> &merged_index) {
        //
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            data_column_type *target_column =
                target.data_containers + cur_column_idx;
            auto old_size = cur_column->data.size();
            auto new_row_size =
                cur_column->data.size() + target_column->data.size();

            cur_column->data.resize(new_row_size);
            cur_column->data.shrink_to_fit();
            thrust::copy(rmm::exec_policy(), target_column->data.begin(),
                         target_column->data.end(),
                         cur_column->data.begin() + old_size);
            target_column->data.resize(0);
            target_column->data.shrink_to_fit();
        }
        // fresh index
        index_container->lex_offset = merged_index;
    }

    // merge two trie assume no duplicates
    void _reorder_merge(self_type &target,
                        rmm::device_vector<col_type> &merged_index) {
        // merge all data column using path merge algorithm

        rmm::device_vector<col_type> tmp_this_col(total_row_size);
        rmm::device_vector<col_type> tmp_target_col(target.total_row_size);
        tmp_target_col.shrink_to_fit();
        tmp_this_col.shrink_to_fit();

        // reorder tuples
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            auto old_size = cur_column->data.size();
            data_column_type *target_column =
                target.data_containers + cur_column_idx;
            cur_column->data.swap(tmp_this_col);
            cur_column->data.resize(merged_index.size());
            thrust::transform(rmm::exec_policy(), merged_index.begin(),
                              merged_index.end(), cur_column->data.begin(),
                              [raw_this_ptr = cur_column->data.begin(),
                               tmp_this_ptr = tmp_this_col.begin(),
                               raw_target_ptr = target_column->data.begin(),
                               old_size] __device__(col_type data) {
                                  if (data < raw_this_ptr.size()) {
                                      return raw_this_ptr[data];
                                  } else {
                                      return raw_target_ptr[data - old_size];
                                  }
                              });
            cur_column->data.shrink_to_fit();
            // clear target column
            target_column->data.clear();
            target_column->data.shrink_to_fit();
        }
    }

    void build_index() {
        // build index column
        // TODO: support multiple indexing type, now we only use sorted hash
        // join
        _compute_hash_index();
    }

    // find duplicated element in data columns and write them into unique vector
    // not that when calling this function, all column has to be sorted
    void _find_dup(rmm::device_vector<bool> &unique_dedup) {
        rmm::device_vector<col_type> tmp_col(total_row_size);
        tmp_col.shrink_to_fit();
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            // thrust::copy(rmm::exec_policy(), cur_column->data.begin(),
            //              cur_column->data.end(), tmp_col.begin());
            thrust::gather(rmm::exec_policy(),
                           index_container->lex_offset.begin(),
                           index_container->lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            thrust::for_each(
                rmm::exec_policy(), thrust::counting_iterator<col_type>(0),
                thrust::counting_iterator<col_type>(row_size),
                [raw_data_ptr = tmp_col.begin(),
                 unique_dedup_ptr =
                     unique_dedup.begin()] __device__(col_type row_idx) {
                    if (row_idx == 0) {
                        unique_dedup_ptr[row_idx] = true;
                    } else {
                        unique_dedup_ptr[row_idx] =
                            unique_dedup_ptr[row_idx - 1] &&
                            (raw_data_ptr[row_idx] !=
                             raw_data_ptr[row_idx - 1]);
                    }
                });
        }
        // if its not sorted order by natural
        tmp_col.resize(0);
        tmp_col.shrink_to_fit();
        rmm::device_vector<bool> dedup_tmp(total_row_size);
        thrust::gather(rmm::exec_policy(), index_container->lex_offset.begin(),
                       index_container->lex_offset.end(), unique_dedup.begin(),
                       tmp_col.begin());
        unique_dedup.swap(dedup_tmp);
    }

    // NOTE: this remove is in NATRUAL order not LEX order
    void _remove_tuples_in_vec(rmm::device_vector<bool> &unique_dedup) {
        int duplicates = thrust::count(rmm::exec_policy(), unique_dedup.begin(),
                                       unique_dedup.end(), false);
        // remove duplicated data
        // rmm::device_vector<col_type> tmp_col(total_row_size);
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            auto new_end = thrust::remove_if(
                rmm::exec_policy(), cur_column->data.begin(),
                cur_column->data.end(), unique_dedup.begin(),
                [] __device__(bool is_unique) { return !is_unique; });
            cur_column->data.resize(new_end - cur_column->data.begin());
            cur_column->shrink_to_fit();
        }
    }

    // dedup all data column
    void _dedup() {
        // prefix sum unique_dedup
        auto row_size = total_row_size;
        if (row_size <= 1)
            return;
        // rmm::device_vector<col_type> tmp_col(row_size);
        // tmp_col.shrink_to_fit();
        rmm::device_vector<bool> unique_dedup(row_size);
        thrust::fill(rmm::exec_policy(), unique_dedup.begin(),
                     unique_dedup.end(), false);
        _find_dup(unique_dedup);
        _remove_tuples_in_vec(unique_dedup);
    }

    // copy all data from flatten_data to trie
    // NOTE: this won't dedup data, and compress index
    void _load(rmm::device_vector<col_type> &flatten_data, uint64_t row_size,
               rmm::device_vector<col_type> &sorted_raw_row_index) {

        // copy data column
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            // copy sorted value into cur_column.data
            cur_column->data.resize(row_size);
            // cur_column.data.shrink_to_fit();
            // use gather to copy data
            thrust::gather(rmm::exec_policy(), sorted_raw_row_index.begin(),
                           sorted_raw_row_index.end(),
                           flatten_data.begin() + row_size * cur_column_idx,
                           cur_column->data.begin());
        }
        total_row_size = row_size;
        // assume sorted here, refresh lexical order in index
        index_container->lex_offset.resize(total_row_size);
        index_container.shrink_to_fit();
        thrust::sequence(index_container->lex_offset.begin(),
                         index_container->lex_offset.end());
        sort_data_column_flag = true;
    }

    void _compute_hash_index() {
        auto row_size = total_row_size;
        index_container->hashes.resize(row_size);
        index_container->index.resize(row_size);
        int cur_column_idx = arity - 1;
        data_column_type *cur_column = data_containers + cur_column_idx;
        thrust::device_vector<col_type> tmp_col(total_row_size);
        thrust::gather(rmm::exec_policy(), index_container.lex_offset.begin(),
                       index_container.lex_offset.end(),
                       cur_column->data.begin(), tmp_col.begin());
        tmp_col.shrink_to_fit();
        thrust::transform(rmm::exec_policy(), tmp_col.begin(), tmp_col.end(),
                          index_container->hashes.begin(),
                          [] __device__(col_type data) {
                              return murmur_hash3((uint32_t)data);
                          });
        cur_column_idx--;
        for (; cur_column_idx >= 0; cur_column_idx--) {

            cur_column = data_containers[cur_column_idx];
            thrust::gather(rmm::exec_policy(),
                           index_container.lex_offset.begin(),
                           index_container.lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            // compute hash
            thrust::transform(rmm::exec_policy(), tmp_col.begin(),
                              tmp_col.end(), index_container->hashes.begin(),
                              [] __device__(col_type data, uint32_t old_v) {
                                  return hash_combine(
                                      murmur_hash3((uint32_t)data), old_v);
                              });
        }
        index_container->offsets.resize(row_size);
        thrust::sequence(index_container->offsets.begin(),
                         index_container->offsets.end());
        thrust::stable_sort_by_key(
            rmm::exec_policy(), index_container->hashes.begin(),
            index_container->hashes.end(), index_container->offsets.begin());
        index_container->offsets.shrink_to_fit();

        thrust::copy(rmm::exec_policy(), thrust::counting_iterator<col_type>(0),
                     thrust::counting_iterator<col_type>(row_size),
                     index_container->index.begin());
        auto new_end = thrust::unique_by_key(
            rmm::exec_policy(), index_container->hashes.begin(),
            index_container->hashes.end(), index_container->index.begin());
        auto new_size = new_end.first - index_container->hashes.begin();
        index_container->hashes.resize(new_size);
        index_container->index.resize(new_size);
        index_container->hashes.shrink_to_fit();
        index_container->index.shrink_to_fit();
    }
};

template <typename col_type>
void sort_raw_data(rmm::device_vector<col_type> &flatten_data,
                   rmm::device_vector<col_type> &sorted_raw_row_index,
                   int row_size, int column_size) {
    sorted_raw_row_index.resize(row_size);
    sorted_raw_row_index.shrink_to_fit();
    thrust::sequence(sorted_raw_row_index.begin(), sorted_raw_row_index.end());
    rmm::device_vector<col_type> tmp_col(row_size);
    tmp_col.shrink_to_fit();

    for (int cur_column_idx = column_size; cur_column_idx >= 0;
         cur_column_idx--) {
        thrust::gather(rmm::exec_policy(), sorted_raw_row_index.begin(),
                       sorted_raw_row_index.end(),
                       flatten_data.begin() + row_size * cur_column_idx,
                       tmp_col.begin());
        thrust::stable_sort_by_key(rmm::exec_policy(), tmp_col.begin(),
                                   tmp_col.end(), sorted_raw_row_index.begin());
    }
}

__device__ __host__ uint32_t murmur_hash3(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;

    return key;
}

__device__ __host__ uint32_t hash_combine(uint32_t v1, uint32_t v2) {
    return v1 ^ (v2 + 0x9e3779b9 + (v1 << 6) + (v1 >> 2));
}
