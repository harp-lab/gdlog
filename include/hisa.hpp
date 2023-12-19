/**
 * A GPU prefix tree implementation.
 */

#pragma once
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <timer.cuh>

using column_data_type_default = int;
using column_container_type_default =
    rmm::device_vector<column_container_type_default>;
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
    rmm::device_vector<uint32_t> offsets;
};

template <typename container_type> struct DataColumn {
    container_type data;
};

template <typename col_type = column_data_type_default,
          typename container_type = rmm::device_vector<col_type>,
          typename column_index_type = rmm::device_vector<col_type>>
struct HISA {
    using trie_column_type = IndexColumn<container_type, column_index_type>;
    trie_column_type index_container;
    using data_column_type = DataColumn<container_type>;
    data_column_type *data_containers;

    int arity;
    int index_column_size;
    int data_column_size;
    col_type total_row_size;

    bool compress_flag = true;
    bool indexed_flag = true;

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

    // remove tuples which is duplicated in another trie
    void difference(HISA &target) {
        rmm::device_vector<col_type> dedup_tmp(total_row_size);
    }

    // merge two trie and remove duplicated tuples in target
    void merge(HISA &target) {
        // merge all data column using path merge algorithm
        rmm::device_vector<col_type> tmp_this_col(total_row_size);
        rmm::device_vector<col_type> tmp_target_col(target.total_row_size);

        rmm::device_vector<col_type> tmp_this_index(total_row_size);
        rmm::device_vector<col_type> tmp_target_index(target.total_row_size);
        // init idx
        thrust::sequence(tmp_this_index.begin(), tmp_this_index.end());
        thrust::sequence(tmp_target_index.begin(), tmp_target_index.end(),
                         total_row_size);
        // a bit vector halding useless merged tmp value, this just to conform
        // thrust
        rmm::device_vector<bool> useless(total_row_size +
                                         target.total_row_size);
        rmm::device_vector<col_type> merged_index(total_row_size +
                                                  target.total_row_size);

        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers[cur_column_idx];
            data_column_type *target_column =
                target.data_containers[cur_column_idx];
            // copy data to tmp
            thurst::gather(rmm::exec_policy(),
                           tmp_target_index.begin(), tmp_target_index.end(),
                           target_column->data.begin(),
                           tmp_target_col.begin());
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
            thrust::merge_by_key(rmm::exec_policy(), tmp_this_col.begin(),
                                 tmp_this_col.end(), tmp_target_col.begin(),
                                 tmp_target_col.end(), tmp_this_index.begin(),
                                 tmp_target_index.begin(), useless.begin(),
                                 merged_index.begin());
            merged_index.shrink_to_fit();
        }

        // reorder tuples
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers[cur_column_idx];
            auto old_size = cur_column->data.size();
            data_column_type *target_column =
                target.data_containers[cur_column_idx];
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
        _build_hash_index();
    }

    // compress index column using CSR-like format
    void compress_index() {
        for (int cur_column_idx = arity - data_column_size; cur_column_idx >= 0;
             cur_column_idx--) {
            trie_column_type *cur_column = index_containers[cur_column_idx];
            cur_column->size = cur_column->data.size();
            cur_column->index.resize(cur_column->size);
            thrust::copy(rmm::exec_policy(),
                         thrust::counting_iterator<col_type>(0),
                         thrust::counting_iterator<col_type>(cur_column->size),
                         cur_column->index.begin());
            auto new_ends = thrust::unique_by_key(
                rmm::exec_policy(), cur_column->data.begin(),
                cur_column->data.end(), cur_column->index.begin());
            cur_column->data.resize(new_ends.first - cur_column->data.begin());
            cur_column->index.resize(new_ends.first -
                                     cur_column->index.begin());
        }
    }

    // dedup trie, based on unique_dedup bitmap-like vector
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
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers[cur_column_idx];
            // thrust::copy(rmm::exec_policy(), cur_column->data.begin(),
            //              cur_column->data.end(), tmp_col.begin());
            thrust::for_each(
                rmm::exec_policy(), thrust::counting_iterator<col_type>(0),
                thrust::counting_iterator<col_type>(row_size),
                [raw_data_ptr = cur_column->data.begin(),
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
        int duplicates = thrust::count(rmm::exec_policy(), unique_dedup.begin(),
                                       unique_dedup.end(), false);
        // remove duplicated data
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers[cur_column_idx];
            auto new_end = thrust::remove_if(
                rmm::exec_policy(), cur_column->data.begin(),
                cur_column->data.end(), unique_dedup.begin(),
                [] __device__(bool is_unique) { return !is_unique; });
            cur_column->data.resize(new_end - cur_column->data.begin());
            cur_column->shrink_to_fit();
        }
    }

    // copy all data from flatten_data to trie
    // NOTE: this won't dedup data, and compress index
    void _load(rmm::device_vector<col_type> &flatten_data, uint64_t row_size,
               rmm::device_vector<col_type> &sorted_raw_row_index) {

        // copy data column
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers[cur_column_idx];
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
    }

    void _compute_hash_index() {
        auto row_size = total_row_size;
        index_container->hashes.resize(row_size);
        index_container->index.resize(row_size);
        int cur_column_idx = arity - 1;
        data_column_type *cur_column = data_containers[cur_column_idx];
        thrust::transform(rmm::exec_policy(), cur_column->data.begin(),
                          cur_column->data.end(),
                          index_container->hashes.begin(),
                          [] __device__(col_type data) {
                              return murmur_hash3((uint32_t)data);
                          });
        cur_column_idx--;
        for (; cur_column_idx >= 0; cur_column_idx--) {
            cur_column = data_containers[cur_column_idx];
            // compute hash
            thrust::transform(
                rmm::exec_policy(), cur_column->data.begin(),
                cur_column->data.end(), index_container->hashes.begin(),
                [] __device__(col_type data, uint32_t old_v) {
                    return hash_combine(murmur_hash3((uint32_t)data), old_v);
                });
        }
        index_container->offsets.resize(row_size);
        thrust::copy(rmm::exec_policy(), thrust::counting_iterator<col_type>(0),
                     thrust::counting_iterator<col_type>(row_size),
                     index_container->offsets.begin());
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

    // flatten trie to data, data is a vertical table
    // NOTE: please make sure data is large enough
    void flatten(col_type *flatten_data) {}
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
