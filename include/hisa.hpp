/**
 * A GPU prefix tree implementation.
 */

#pragma once
#include <cstdint>
#include <iostream>
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
#include <utility>

#include <thrust/host_vector.h>

#include "../include/hash.h"
#include "../include/timer.cuh"

using column_data_type_default = int;
using index_type_default = uint32_t;
using column_container_type_default =
    rmm::device_vector<column_data_type_default>;
using column_index_type_default = rmm::device_vector<index_type_default>;

template <typename col_type>
void sort_raw_data(rmm::device_vector<col_type> &flatten_data,
                   rmm::device_vector<col_type> &sorted_raw_row_index,
                   int row_size, int column_size) {
    sorted_raw_row_index.resize(row_size);
    sorted_raw_row_index.shrink_to_fit();
    thrust::sequence(sorted_raw_row_index.begin(), sorted_raw_row_index.end());
    rmm::device_vector<col_type> tmp_col(row_size);
    tmp_col.shrink_to_fit();

    for (int cur_column_idx = column_size - 1; cur_column_idx >= 0;
         cur_column_idx--) {
        thrust::gather(rmm::exec_policy(), sorted_raw_row_index.begin(),
                       sorted_raw_row_index.end(),
                       flatten_data.begin() + row_size * cur_column_idx,
                       tmp_col.begin());
        thrust::stable_sort_by_key(rmm::exec_policy(), tmp_col.begin(),
                                   tmp_col.end(), sorted_raw_row_index.begin());
    }
}

template <typename T> void print_device_vector(rmm::device_vector<T> &vec) {
    thrust::host_vector<T> vec_host(vec.begin(), vec.end());
    for (auto i : vec_host) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

namespace compare {
template <size_t I = 0, typename T, typename... Ts>
__host__ __device__ constexpr bool
tuple_less(const thrust::tuple<T, Ts...> &lhs,
           const thrust::tuple<T, Ts...> &rhs) {
    if (I < sizeof...(Ts)) {
        return false;
    }
    if (thrust::get<I>(lhs) < thrust::get<I>(rhs)) {
        return true;
    } else if (thrust::get<I>(lhs) == thrust::get<I>(rhs)) {
        return tuple_less<I + 1>(lhs, rhs);
    } else {
        return false;
    }
}

template <size_t I = 0, typename T, typename... Ts>
__host__ __device__ constexpr bool
tuple_equal(const thrust::tuple<T, Ts...> &lhs,
            const thrust::tuple<T, Ts...> &rhs) {
    if (I < sizeof...(Ts)) {
        return true;
    }
    if (thrust::get<I>(lhs) == thrust::get<I>(rhs)) {
        return tuple_equal<I + 1>(lhs, rhs);
    } else {
        return false;
    }
}
} // namespace compare
// a variadic comparator for thrust::tuple
struct tuple_comparator {
    template <typename... Ts>
    __host__ __device__ bool operator()(const thrust::tuple<Ts...> &lhs,
                                        const thrust::tuple<Ts...> &rhs) {
        return compare::tuple_less(lhs, rhs);
    }
};

template <typename container_type, typename index_container_type>
struct IndexColumn {
    // uint64_t size;
    // all value in current column, this vector is unique
    // container_type data;
    // need a better hash map?
    index_container_type hashes;
    // CSR index based on hashes;
    container_type index;
    // tuple permutation in hash order
    container_type offsets;
    // tuple permutation in lexical order
    container_type lex_offset;
};

template <typename container_type> struct DataColumn {
    container_type data;
};

template <typename Array, std::size_t... I>
auto iter_tuple_begin(const Array &a, std::index_sequence<I...>) {
    return thrust::make_tuple(a[I].data.begin()...);
}
template <typename Array, std::size_t... I>
auto iter_tuple_end(const Array &a, std::index_sequence<I...>) {
    return thrust::make_tuple(a[I].data.end()...);
}
// a function zip n vectors
template <size_t N, typename T, typename Indices = std::make_index_sequence<N>>
auto zip_iter_n(DataColumn<T> *col) {
    return thrust::make_zip_iterator(iter_tuple_begin(col, Indices{}));
}

template <typename col_type = column_data_type_default,
          typename column_index_type = index_type_default,
          typename col_size_type = col_type,
          typename data_container_type = rmm::device_vector<col_type>,
          typename index_container_type = rmm::device_vector<column_index_type>,
          typename col_size_container_type = rmm::device_vector<col_size_type>>
struct HISA {
    using index_column_type =
        IndexColumn<data_container_type, index_container_type>;
    index_column_type *index_container;
    using data_column_type = DataColumn<data_container_type>;
    data_column_type *data_containers;

    using self_type =
        HISA<col_type, column_index_type, col_size_type, data_container_type,
             index_container_type, col_size_container_type>;

    size_t arity;
    int index_column_size;
    int data_column_size;
    col_type total_row_size;

    constexpr static float load_factor = 0.8f;

    bool compress_flag = true;
    bool indexed_flag = true;

    // this flag indicate if row order in each data columns
    // follow the index ording
    bool sort_data_column_flag = true;

    HISA(int arity, int index_column_size, bool compress_flag = true)
        : arity(arity), index_column_size(index_column_size),
          compress_flag(compress_flag) {
        index_container = new index_column_type;
        data_containers = new data_column_type[arity];
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
        timer.start_timer();
        rmm::device_vector<col_type> sorted_raw_row_index(row_size);
        sort_raw_data(flatten_data, sorted_raw_row_index, row_size, arity);
        timer.stop_timer();
        detail_time[0] += timer.get_spent_time();

        timer.start_timer();
        _load(flatten_data, row_size, sorted_raw_row_index);
        timer.stop_timer();
        detail_time[1] += timer.get_spent_time();

        timer.start_timer();
        dedup();
        timer.stop_timer();
        detail_time[2] += timer.get_spent_time();
        // std::cout << "dedup data" << std::endl;

        // remove from lex offset

        if (indexed_flag) {
            timer.start_timer();
            build_index();
            timer.stop_timer();
            detail_time[3] += timer.get_spent_time();
        }
    }

    void merge_unique(self_type &target) {
        // merge using set union
        auto old_size_before_merge = total_row_size;
        _unordered_merge(target);
        index_container->lex_offset.resize(total_row_size +
                                           target.total_row_size);
        thrust::sequence(
            index_container->lex_offset.begin() + old_size_before_merge,
            index_container->lex_offset.end(), old_size_before_merge);
        total_row_size += target.total_row_size;
        resort();
    }

    // merge and deduplicates, leave set_differece tuples in target
    // need rebuild index for target after use this function
    void difference_and_merge(self_type &target) {
        auto old_size_before_merge = total_row_size;
        rmm::device_vector<col_type> merged_index(total_row_size +
                                                  target.total_row_size);
        merged_index.shrink_to_fit();

        // _reorder_merge(target, merged_index);
        _unordered_merge(target);
        // std::cout << "extend data >>>>>>>>>> " << std::endl;
        _merge_permutation(target, merged_index);
        total_row_size = merged_index.size();
        index_container->lex_offset.swap(merged_index);
        merged_index.resize(0);
        merged_index.shrink_to_fit();
        print_tuples("after merge without dedup");

        // std::cout << "merge permutation >>>>>>>>>> " << std::endl;;
        rmm::device_vector<bool> unique_dedup(
            index_container->lex_offset.size());
        thrust::fill(rmm::exec_policy(), unique_dedup.begin(),
                     unique_dedup.end(), false);
        _find_dup(unique_dedup);
        print_device_vector(unique_dedup);
        int duplicates = _remove_tuples_in_vec(unique_dedup);
        // drop all dups in permutation
        auto new_end_unique_dedup = thrust::remove_if(
            rmm::exec_policy(), index_container->lex_offset.begin(),
            index_container->lex_offset.end(), unique_dedup.begin(),
            [] __device__(bool is_unique) { return !is_unique; });
        auto new_perm_size =
            new_end_unique_dedup - index_container->lex_offset.begin();
        index_container->lex_offset.resize(new_perm_size);

        int new_target_size = target.total_row_size - duplicates;
        rmm::device_vector<col_type> row_index_perm(total_row_size);
        thrust::sequence(row_index_perm.begin(), row_index_perm.end());
        auto new_end = thrust::remove_if(
            rmm::exec_policy(), row_index_perm.begin(), row_index_perm.end(),
            [ptr = index_container->lex_offset.data().get(),
             total_row_size = old_size_before_merge] __device__(col_type a) {
                return ptr[a] < total_row_size;
            });
        row_index_perm.resize(new_end - row_index_perm.begin());
        row_index_perm.shrink_to_fit();
        auto perm_begin = thrust::make_permutation_iterator(
            index_container->lex_offset.begin(), row_index_perm.begin());
        thrust::sequence(perm_begin, perm_begin + new_target_size,
                         old_size_before_merge);

        // copy back to target
        for (int i = arity - 1; i >= 0; i--) {
            // need faster copy
            auto cur_column = data_containers + i;
            auto target_column = target.data_containers + i;
            target_column->data.resize(new_target_size);
            thrust::copy(cur_column->data.begin() + old_size_before_merge,
                         cur_column->data.end(), target_column->data.begin());
        }
        target.total_row_size = new_target_size;
        target.index_container->lex_offset.resize(new_target_size);
        thrust::sequence(target.index_container->lex_offset.begin(),
                         target.index_container->lex_offset.end());
        // target.build_index();`
    }

    // merge and store merged permutation in merged_index
    // in permutation, 0 ~ total_row_size comes from `this`,
    // total_row_size ~ total_row_size + target->total_row_size comes from
    // target
    void _merge_permutation(self_type &target,
                            rmm::device_vector<col_type> &merged_index) {
        // merge all data column using path merge algorithm
        auto tmp_target_index_begin =
            thrust::make_counting_iterator(total_row_size);
        auto tmp_target_index_end =
            tmp_target_index_begin + target.total_row_size;
        auto this_index_begin = thrust::make_counting_iterator(0);
        auto this_index_end = this_index_begin + total_row_size;

        merged_index.resize(total_row_size + target.total_row_size);
        thrust::device_vector<col_type *> data_containers_ptr(arity);
        for (int i = 0; i < arity; i++) {
            data_containers_ptr[i] = data_containers[i].data.data().get();
        }

        auto merge_end = thrust::merge(
            rmm::exec_policy(), this_index_begin, this_index_end,
            tmp_target_index_begin, tmp_target_index_end, merged_index.begin(),
            [data_containers = data_containers_ptr.data().get(),
             arity = arity] __device__(col_type a, col_type b) -> bool {
                for (int i = 0; i < arity; i++) {
                    col_type *cur_column = data_containers[i];
                    if (cur_column[a] != cur_column[b]) {
                        return cur_column[a] < cur_column[b];
                        ;
                    }
                }
                return false;
            });

        merged_index.resize(merge_end - merged_index.begin());
        merged_index.shrink_to_fit();
    }

    // this merge will not change any data, but just merge them directly
    // put target at the end
    void _unordered_merge(self_type &target) {
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
            // clear target column
            target_column->data.resize(0);
            target_column->data.shrink_to_fit();
        }
        // fresh index
    }

    void build_index() {
        // build index column
        // TODO: support multiple indexing type, now we only use sorted hash
        // join
        _compute_hash_index();
    }

    // find duplicated element in data columns and write them into unique
    // vector not that when calling this function, all column has to be
    // sorted
    void _find_dup(rmm::device_vector<bool> &unique_dedup) {
        rmm::device_vector<col_type> tmp_col(total_row_size);
        tmp_col.shrink_to_fit();
        unique_dedup[0] = true;
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            thrust::gather(rmm::exec_policy(),
                           index_container->lex_offset.begin(),
                           index_container->lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            rmm::device_vector<bool> dedup_tmp(total_row_size);
            // print_device_vector(tmp_col);
            dedup_tmp[0] = true;
            thrust::transform(rmm::exec_policy(), tmp_col.begin(),
                              tmp_col.end() - 1, tmp_col.begin() + 1,
                              dedup_tmp.begin() + 1,
                              [] __device__(col_type data_l, col_type data_r) {
                                  return data_l != data_r;
                              });
            // print_device_vector(dedup_tmp);
            thrust::transform(rmm::exec_policy(), unique_dedup.begin(),
                              unique_dedup.end(), dedup_tmp.begin(),
                              unique_dedup.begin(),
                              [] __device__(bool a, bool b) { return a || b; });
            // std::cout << "find dup " << cur_column_idx << std::endl;
        }
    }

    // NOTE: this remove is in NATRUAL order not LEX order
    int _remove_tuples_in_vec(rmm::device_vector<bool> &unique_dedup) {
        int duplicates = thrust::count(rmm::exec_policy(), unique_dedup.begin(),
                                       unique_dedup.end(), false);
        // remove duplicated data
        rmm::device_vector<bool> unique_dedup_tmp(unique_dedup.size());

        thrust::for_each(
            rmm::exec_policy(),
            thrust::make_zip_iterator(thrust::make_tuple(
                index_container->lex_offset.begin(), unique_dedup.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                index_container->lex_offset.end(), unique_dedup.end())),
            [tmp_ptr = unique_dedup_tmp.data().get()] __device__(
                thrust::tuple<col_type, bool> t) {
                tmp_ptr[thrust::get<0>(t)] = thrust::get<1>(t);
            });
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            auto new_end = thrust::remove_if(
                rmm::exec_policy(), cur_column->data.begin(),
                cur_column->data.end(), unique_dedup_tmp.begin(),
                [] __device__(bool is_unique) { return !is_unique; });
            cur_column->data.resize(new_end - cur_column->data.begin());
            std::cout << "col after remove " << cur_column->data.size()
                      << std::endl;
            cur_column->data.shrink_to_fit();
        }
        auto new_total_row_size = total_row_size - duplicates;
        total_row_size = new_total_row_size;
        return duplicates;
    }

    // dedup all data column
    void dedup() {
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

        // assume this only called when data is sorted
        index_container->lex_offset.resize(total_row_size);
        thrust::sequence(index_container->lex_offset.begin(),
                         index_container->lex_offset.end());
        index_container->lex_offset.shrink_to_fit();
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
        index_container->lex_offset.shrink_to_fit();
        thrust::sequence(index_container->lex_offset.begin(),
                         index_container->lex_offset.end());
        sort_data_column_flag = true;
    }

    void all_tuple_hashes(index_container_type &hashes) {
        auto row_size = total_row_size;
        hashes.resize(row_size);
        hashes.shrink_to_fit();
        int cur_column_idx = arity - 1;
        data_column_type *cur_column = data_containers + cur_column_idx;
        rmm::device_vector<col_type> tmp_col(total_row_size);
        thrust::gather(rmm::exec_policy(), index_container->lex_offset.begin(),
                       index_container->lex_offset.end(),
                       cur_column->data.begin(), tmp_col.begin());
        tmp_col.shrink_to_fit();
        thrust::transform(rmm::exec_policy(), tmp_col.begin(), tmp_col.end(),
                          hashes.begin(),
                          [] __device__(col_type data) {
                              return murmur_hash3((uint32_t)data);
                          });
        cur_column_idx--;
        for (; cur_column_idx >= 0; cur_column_idx--) {
            cur_column = data_containers + cur_column_idx;
            thrust::gather(rmm::exec_policy(),
                           index_container->lex_offset.begin(),
                           index_container->lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            // compute hash
            thrust::transform(
                rmm::exec_policy(), hashes.begin(), hashes.end(), tmp_col.begin(),
                hashes.begin(),
                [] __device__(uint32_t old_v, col_type data) {
                    return hash_combine(murmur_hash3((uint32_t)data), old_v);
                });
        }
    }

    void _compute_hash_index() {
        auto row_size = total_row_size;
        all_tuple_hashes(index_container->hashes);
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
        // create a tmp for hash
        rmm::device_vector<col_type> tmp_hash(row_size);
        tmp_hash.shrink_to_fit();
        thrust::copy(rmm::exec_policy(), index_container->hashes.begin(),
                     index_container->hashes.end(), tmp_hash.begin());
        auto new_end = thrust::unique_by_key(
            rmm::exec_policy(), tmp_hash.begin(),
            tmp_hash.end(), index_container->index.begin());
        auto new_size = new_end.first - index_container->index.begin();
        index_container->index.resize(new_size);
        index_container->index.shrink_to_fit();
    }

    
    void resort() {
        // resort index permutation
        rmm::device_vector<col_type> tmp_col(total_row_size);
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            // copy sorted value into cur_column.data
            // use gather to copy data
            thrust::gather(rmm::exec_policy(),
                           index_container->lex_offset.begin(),
                           index_container->lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            thrust::stable_sort_by_key(rmm::exec_policy(), tmp_col.begin(),
                                       tmp_col.end(),
                                       index_container->lex_offset.begin());
        }
        // reorder data column based on lex_offset
        for (int cur_column_idx = arity - 1; cur_column_idx >= 0;
             cur_column_idx--) {
            data_column_type *cur_column = data_containers + cur_column_idx;
            // copy sorted value into cur_column.data
            // use gather to copy data
            thrust::gather(rmm::exec_policy(),
                           index_container->lex_offset.begin(),
                           index_container->lex_offset.end(),
                           cur_column->data.begin(), tmp_col.begin());
            cur_column->data.swap(tmp_col);
        }
        thrust::sequence(index_container->lex_offset.begin(),
                         index_container->lex_offset.end());
    }

    void reload() {
        resort();
        dedup();
        if (indexed_flag) {
            build_index();
        }
    }

    void print_tuples(char *msg) {
        std::cout << "HISA >>> " << msg << std::endl;
        std::cout << "Total tuples counts:  " << total_row_size << std::endl;
        // copy to host
        thrust::host_vector<col_type> host_indices(total_row_size);
        thrust::copy(index_container->lex_offset.begin(),
                     index_container->lex_offset.end(), host_indices.begin());
        thrust::host_vector<col_type> *host_cols =
            new thrust::host_vector<col_type>[arity];

        for (int i = 0; i < arity; i++) {
            host_cols[i].resize(total_row_size);
            thrust::copy(data_containers[i].data.begin(),
                         data_containers[i].data.end(), host_cols[i].begin());
        }

        for (auto &idx : host_indices) {
            std::cout << idx << ":\t";
            for (int i = 0; i < arity; i++) {
                std::cout << host_cols[i][idx] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << "end <<<" << std::endl;

        delete[] host_cols;
    }
};

template <typename col_type, typename column_index_type>
void print_hisa(HISA<col_type, column_index_type> &hisa, char *msg) {
    std::cout << "HISA >>> " << msg << std::endl;
    std::cout << "Total tuples counts:  " << hisa.total_row_size << std::endl;
    // copy to host
    thrust::host_vector<col_type> host_indices(hisa.total_row_size);
    thrust::copy(hisa.index_container->lex_offset.begin(),
                 hisa.index_container->lex_offset.end(), host_indices.begin());
    thrust::host_vector<col_type> *host_cols =
        new thrust::host_vector<col_type>[hisa.arity];

    for (int i = 0; i < hisa.arity; i++) {
        host_cols[i].resize(hisa.total_row_size);
        // host_cols[i] = hisa.data_containers[i].data;
        thrust::copy(hisa.data_containers[i].data.begin(),
                     hisa.data_containers[i].data.end(), host_cols[i].begin());
    }

    for (auto &idx : host_indices) {
        std::cout << idx << ":\t";
        for (int i = 0; i < hisa.arity; i++) {
            std::cout << host_cols[i][idx] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "end <<<" << std::endl;

    delete[] host_cols;
}
