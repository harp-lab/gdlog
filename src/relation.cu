
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relation.cuh"
#include "../include/timer.cuh"
#include "../include/tuple.cuh"
#include <chrono>
#include <iostream>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <rmm/exec_policy.hpp>

__global__ void calculate_index_hash(tuple_type *tuples,
                                     tuple_size_t tuple_counts,
                                     int index_column_size, MEntity *index_map,
                                     tuple_size_t index_map_size,
                                     tuple_indexed_less cmp) {
    tuple_size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts)
        return;

    tuple_size_t stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < tuple_counts; i += stride) {
        tuple_type cur_tuple = tuples[i];

        u64 hash_val = prefix_hash(cur_tuple, index_column_size);
        u64 request_hash_index = hash_val % index_map_size;
        tuple_size_t position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(index_map[position].key),
                                         EMPTY_HASH_ENTRY, hash_val);
            tuple_size_t existing_value = index_map[position].value;
            if (existing_key == EMPTY_HASH_ENTRY || existing_key == hash_val) {
                bool collison_flag = false;
                while (true) {
                    if (existing_value < i) {
                        // occupied entry, but no need for swap, just check if
                        // collision
                        if (!tuple_eq(tuples[existing_value], cur_tuple,
                                      index_column_size)) {
                            // collision, find nex available entry
                            collison_flag = true;
                            break;
                        } else {
                            // no collision but existing tuple is smaller, in
                            // this case, not need to swap, just return(break;
                            // break)
                            break;
                        }
                    }
                    if (existing_value > i &&
                        existing_value != EMPTY_HASH_ENTRY) {
                        // occupied entry, may need for swap
                        if (!tuple_eq(tuples[existing_value], cur_tuple,
                                      index_column_size)) {
                            // collision, find nex available entry
                            collison_flag = true;
                            break;
                        }
                        // else, swap
                    }
                    // swap value
                    if (existing_value == i) {
                        // swap success return
                        break;
                    } else {
                        // need swap
                        existing_value = atomicCAS(&(index_map[position].value),
                                                   existing_value, i);
                    }
                }
                if (!collison_flag) {
                    break;
                }
            }
            position = (position + 1) % index_map_size;
        }
    }
}

__global__ void init_tuples_unsorted(tuple_type *tuples, column_type *raw_data,
                                     int arity, tuple_size_t rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        tuples[i] = raw_data + i * arity;
    }
}

void init_tuples_unsorted_thrust(tuple_type *tuples, column_type *raw_data,
                                 int arity, tuple_size_t rows) {
    thrust::counting_iterator<tuple_size_t> index_sequence_begin(0);
    thrust::counting_iterator<tuple_size_t> index_sequence_end(rows);
    thrust::transform(rmm::exec_policy(),
        index_sequence_begin,
        index_sequence_end,
        tuples,
        [raw_data, arity] __device__(tuple_size_t index) {
            return raw_data + index * arity;
        });
}

__global__ void
get_join_result_size(MEntity *inner_index_map,
                     tuple_size_t inner_index_map_size,
                     tuple_size_t inner_tuple_counts, tuple_type *inner_tuples,
                     tuple_type *outer_tuples, tuple_size_t outer_tuple_counts,
                     int outer_index_column_size, int join_column_counts,
                     tuple_generator_hook tp_gen, tuple_predicate tp_pred,
                     tuple_size_t *join_result_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_tuples[i];

        tuple_size_t current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_index_map[index_position].key == hash_val &&
                tuple_eq(outer_tuple,
                         inner_tuples[inner_index_map[index_position].value],
                         outer_index_column_size)) {
                break;
            } else if (inner_index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_index_map_size;
        }
        if (index_not_exists) {
            continue;
        }
        // pull all joined elements
        tuple_size_t position = inner_index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_tuples[position];
            bool cmp_res = tuple_eq(inner_tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // hack to apply filter
                // TODO: this will cause max arity of a relation is 20
                if (tp_gen != nullptr && tp_pred != nullptr) {
                    column_type tmp[20] = {0};
                    (*tp_gen)(cur_inner_tuple, outer_tuple, tmp);
                    if ((*tp_pred)(tmp)) {
                        current_size++;
                    }
                } else {
                    current_size++;
                }
            } else {
                break;
            }
            position = position + 1;
            if (position > inner_tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
    }
}

__global__ void
get_join_result(MEntity *inner_index_map, tuple_size_t inner_index_map_size,
                tuple_size_t inner_tuple_counts, tuple_type *inner_tuples,
                tuple_type *outer_tuples, tuple_size_t outer_tuple_counts,
                int outer_index_column_size, int join_column_counts,
                tuple_generator_hook tp_gen, tuple_predicate tp_pred,
                int output_arity, column_type *output_raw_data,
                tuple_size_t *res_count_array, tuple_size_t *res_offset,
                JoinDirection direction) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        tuple_type outer_tuple = outer_tuples[i];

        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_index_map[index_position].key == hash_val &&
                tuple_eq(outer_tuple,
                         inner_tuples[inner_index_map[index_position].value],
                         outer_index_column_size)) {
                break;
            } else if (inner_index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_index_map_size;
        }
        if (index_not_exists) {
            continue;
        }

        // pull all joined elements
        tuple_size_t position = inner_index_map[index_position].value;
        while (true) {
            // TODO: always put join columns ahead? could be various benefits
            // but memory is issue to mantain multiple copies
            bool cmp_res = tuple_eq(inner_tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // tuple prefix match, join here
                tuple_type inner_tuple = inner_tuples[position];
                tuple_type new_tuple =
                    output_raw_data +
                    (res_offset[i] + current_new_tuple_cnt) * output_arity;

                // for (int j = 0; j < output_arity; j++) {
                // TODO: this will cause max arity of a relation is 20
                if (tp_gen != nullptr && tp_pred != nullptr) {
                    column_type tmp[20];
                    (*tp_gen)(inner_tuple, outer_tuple, tmp);
                    // printf("tmp: %d %d\n", tmp[0], tmp[1]);
                    if ((*tp_pred)(tmp)) {
                        (*tp_gen)(inner_tuple, outer_tuple, new_tuple);
                        current_new_tuple_cnt++;
                    }
                } else {
                    (*tp_gen)(inner_tuple, outer_tuple, new_tuple);
                    current_new_tuple_cnt++;
                }
                if (current_new_tuple_cnt > res_count_array[i]) {
                    break;
                }
            } else {
                // bucket end
                break;
            }
            position = position + 1;
            if (position > (inner_tuple_counts - 1)) {
                // end of data arrary
                break;
            }
        }
    }
}

__global__ void flatten_tuples_raw_data(tuple_type *tuple_pointers,
                                        column_type *raw,
                                        tuple_size_t tuple_counts, int arity) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < tuple_counts; i += stride) {
        for (int j = 0; j < arity; j++) {
            raw[i * arity + j] = tuple_pointers[i][j];
        }
    }
}

struct update_functor {
    tuple_type *tuple_pointers;
    column_type *raw;
    int arity;
    __host__ __device__ void operator()(int index) {
        for (int j = 0; j < arity; j++) {
            raw[index * arity + j] = tuple_pointers[index][j];
        }
    }
};

void flatten_tuples_raw_data_thrust(tuple_type *tuple_pointers,
                                    column_type *raw, tuple_size_t tuple_counts,
                                    int arity) {
    thrust::counting_iterator<tuple_size_t> index_sequence_begin(0);
    thrust::counting_iterator<tuple_size_t> index_sequence_end(tuple_counts);
    thrust::for_each(rmm::exec_policy(), index_sequence_begin, index_sequence_end,
                     update_functor{tuple_pointers, raw, arity});
}

__global__ void get_copy_result(tuple_type *src_tuples,
                                column_type *dest_raw_data, int output_arity,
                                tuple_size_t tuple_counts,
                                tuple_copy_hook tp_gen) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < tuple_counts; i += stride) {
        tuple_type dest_tp = dest_raw_data + output_arity * i;
        (*tp_gen)(src_tuples[i], dest_tp);
    }
}

void Relation::flush_delta(int grid_size, int block_size, float *detail_time) {
    if (delta->tuple_counts == 0) {
        return;
    }
    KernelTimer timer;
    timer.start_timer();
    tuple_size_t new_full_size = full->tuple_counts + delta->tuple_counts;

    bool extened_mem = false;

    tuple_size_t total_mem_size = get_total_memory();
    tuple_size_t free_mem = get_free_memory();
    u64 delta_mem_size = delta->tuple_counts * sizeof(tuple_type);
    int multiplier = FULL_BUFFER_VEC_MULTIPLIER;
    if (!pre_allocated_merge_buffer_flag && !fully_disable_merge_buffer_flag &&
        delta_mem_size * multiplier <= 0.1 * free_mem) {
        // std::cout << "reenable pre-allocated merge buffer" << std::endl;
        pre_allocated_merge_buffer_flag = true;
    }

    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        if (tuple_merge_buffer_size <= new_full_size) {
            // resize buffer to make sure **next** swap with full, swapped
            // buffer still has size tuple_merge_buffer_size
            // std::cout << "extend mem" << std::endl;
            extened_mem = true;
            tuple_merge_buffer_size =
                full->tuple_counts + (delta->tuple_counts * multiplier);
            u64 tuple_full_buf_mem_size =
                tuple_merge_buffer_size * sizeof(tuple_type);

            while (((free_mem - tuple_full_buf_mem_size) * 1.0 /
                    total_mem_size) < 0.4 &&
                   delta_mem_size * multiplier > 0.1 * free_mem) {
                // std::cout << "multiplier : " << multiplier << std::endl;
                multiplier--;
                tuple_merge_buffer_size =
                    full->tuple_counts + (delta->tuple_counts * multiplier);
                tuple_full_buf_mem_size =
                    tuple_merge_buffer_size * sizeof(tuple_type);
                if (multiplier == 2) {
                    // std::cout << "not enough memory for merge buffer"
                            //   << std::endl;
                    // not enough space for pre-allocated buffer
                    pre_allocated_merge_buffer_flag = false;
                    tuple_merge_buffer_size = new_full_size;
                    // cudaFree(tuple_merge_buffer);
                    tuple_merge_buffer.resize(new_full_size);
                    break;
                }
            }
            if (pre_allocated_merge_buffer_flag) {
                tuple_merge_buffer.resize(tuple_merge_buffer_size);
            }
        }
        // else no need resize buffer
    } else {
        tuple_merge_buffer_size = new_full_size;
        tuple_merge_buffer.resize(tuple_merge_buffer_size);
    }
    // std::cout << new_full_size << std::endl;
    // TODO:
    timer.stop_timer();
    // std::cout << "malloc time : " << timer.get_spent_time() << std::endl;
    detail_time[0] = timer.get_spent_time();

    timer.start_timer();
    thrust::merge(
        rmm::exec_policy(), full->tuples.begin(),
        full->tuples.begin() + full->tuple_counts, delta->tuples.begin(),
        delta->tuples.begin() + delta->tuple_counts, tuple_merge_buffer.begin(),
        tuple_indexed_less(delta->index_column_size, delta->arity));
    // swap buffer and full tuples after merged
    timer.stop_timer();
    // std::cout << "merge time : " << timer.get_spent_time() << std::endl;
    detail_time[1] = timer.get_spent_time();

    timer.start_timer();
    full->tuples.swap(tuple_merge_buffer);
    full->tuple_counts = new_full_size;
    // full->tuples.resize(new_full_size);
    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        if (extened_mem) {
            // if extended size of merge buffer, after swap, the buffer will
            // become original full, which has less size than merge_buffer_size
            // so we need to resize it
            tuple_merge_buffer.resize(tuple_merge_buffer_size);
        }
    } else {
        // if not using merge buffer, we need to clean buffer
        // tuple_merge_buffer.clear();
    }
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
    buffered_delta_vectors.push_back(delta);
    if (index_flag) {
        timer.start_timer();
        full->update_index_map(grid_size, block_size);
        timer.stop_timer();
        detail_time[3] = timer.get_spent_time();
    } else {
        detail_time[3] = 0;
    }
}

void GHashRelContainer::update_index_map(int grid_size, int block_size, float load_factor) {
    // init the index map
    // set the size of index map same as data, (this should give us almost
    // no conflict) however this can be memory inefficient
    index_map_load_factor = load_factor;
    index_map_size = std::ceil(tuple_counts / index_map_load_factor);
    // target->index_map_size = data_row_size;
    index_map.resize(index_map_size);
    thrust::fill(rmm::exec_policy(), index_map.begin(), index_map.end(),
                 MEntity{EMPTY_HASH_ENTRY, EMPTY_HASH_ENTRY});

    // print_tuple_rows(this, "before update index map");
    // print_hashes(this, "before update index map");
    // calculate hash
    calculate_index_hash<<<grid_size, block_size>>>(
        tuples.data().get(), tuple_counts, index_column_size,
        index_map.data().get(), index_map_size,
        tuple_indexed_less(index_column_size, arity));
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
}

void copy_relation_container(GHashRelContainer *dst, GHashRelContainer *src,
                             int grid_size, int block_size) {
    dst->index_map_size = src->index_map_size;
    dst->index_map_load_factor = src->index_map_load_factor;
    dst->index_column_size = src->index_column_size;
    dst->tuple_counts = src->tuple_counts;
    dst->data_raw_row_size = src->data_raw_row_size;
    dst->arity = src->arity;
    dst->dependent_column_size = src->dependent_column_size;
    // dst->data_raw = src->data_raw;
    dst->data_raw.resize(src->tuple_counts * src->arity);
    dst->tuples.resize(src->tuple_counts);
    flatten_tuples_raw_data_thrust(src->tuples.data().get(),
                                   dst->data_raw.data().get(),
                                   src->tuple_counts, src->arity);
    init_tuples_unsorted<<<grid_size, block_size>>>(
        dst->tuples.data().get(), dst->data_raw.data().get(), src->arity,
        src->tuple_counts);
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
    thrust::sort(rmm::exec_policy(), dst->tuples.begin(),
                 dst->tuples.begin() + src->tuple_counts,
                 tuple_indexed_less(src->index_column_size, src->arity));
    dst->update_index_map(grid_size, block_size);
}

void free_relation_container(GHashRelContainer *target) {
    target->tuple_counts = 0;
    target->index_map_size = 0;
    target->data_raw_row_size = 0;
    target->index_map.clear();
    target->index_map.shrink_to_fit();
    target->tuples.clear();
    target->tuples.shrink_to_fit();
    target->data_raw.clear();
    target->data_raw.shrink_to_fit();
}

void load_relation(Relation *target, std::string name, int arity,
                   column_type *data, tuple_size_t data_row_size,
                   tuple_size_t index_column_size, int dependent_column_size,
                   int grid_size, int block_size, bool tmp_flag) {

    target->name = name;
    target->arity = arity;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    target->tmp_flag = tmp_flag;
    target->full =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    target->delta =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    target->newt =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    // target->newt->tmp_flag = tmp_flag;

    float detail_time[5];
    // everything must have a full
    thrust::host_vector<column_type> tmp(data_row_size * arity);
    tmp.assign(data, data + data_row_size * arity);
    target->full->data_raw = tmp;
    target->full->data_raw_row_size = data_row_size;
    target->full->tuple_counts = data_row_size;
    target->full->tuples.resize(data_row_size);
    init_tuples_unsorted<<<grid_size, block_size>>>(
        target->full->tuples.data().get(), target->full->data_raw.data().get(),
        arity, data_row_size);
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaGetLastError());
    thrust::sort(rmm::exec_policy(), target->full->tuples.begin(),
                 target->full->tuples.begin() + data_row_size,
                 tuple_indexed_less(index_column_size, arity));
    target->full->update_index_map(grid_size, block_size);
}

// __global__ void find_duplicate_tuples(GHashRelContainer *target,
//                                       tuple_type *new_tuples,
//                                       tuple_size_t new_tuple_counts,
//                                       bool *duplicate_bitmap,
//                                       tuple_size_t *duplicate_counts) {
//     tuple_size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (index >= new_tuple_counts)
//         return;

//     tuple_size_t stride = blockDim.x * gridDim.x;
//     for (tuple_size_t i = index; i < new_tuple_counts; i += stride) {
//         tuple_type cur_tuple = new_tuples[i];

//         u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
//         tuple_size_t position = hash_val % target->index_map_size;
//         u64 existing_key = target->index_map[position].key;
//         auto cur_target_pos = target->index_map[position].value;
//         if (existing_key == EMPTY_HASH_ENTRY) {
//             continue;
//         }
//         if (cur_target_pos == EMPTY_HASH_ENTRY) {
//             continue;
//         }
//         // if (existing_key == hash_val) {
//         while (true) {
//             auto cur_target_tuple = target->tuples[cur_target_pos];

//             // printf("cur_target_tuple: %d %d\n", cur_target_tuple[0],
//             // cur_target_tuple[1]);
//             if (hash_val != prefix_hash(cur_target_tuple, 1)) {
//                 break;
//             }
//             if (tuple_eq(cur_tuple, cur_target_tuple, target->arity)) {
//                 // duplicate_bitmap[i] = true;
//                 // atomicAdd(duplicate_counts, 1);
//                 new_tuples[i] = nullptr;
//                 break;
//             }
//             cur_target_pos = (cur_target_pos + 1) % target->tuple_counts;
//         }
//         // }
//     }
// }
