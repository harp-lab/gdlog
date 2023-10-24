
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relation.cuh"
#include "../include/timer.cuh"
#include "../include/tuple.cuh"
#include <chrono>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/unique.h>

__global__ void calculate_index_hash(GHashRelContainer *target,
                                     tuple_indexed_less cmp) {
    tuple_size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->tuple_counts)
        return;

    tuple_size_t stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < target->tuple_counts; i += stride) {
        tuple_type cur_tuple = target->tuples[i];

        u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
        u64 request_hash_index = hash_val % target->index_map_size;
        tuple_size_t position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(target->index_map[position].key),
                                         EMPTY_HASH_ENTRY, hash_val);
            tuple_size_t existing_value = target->index_map[position].value;
            if (existing_key == EMPTY_HASH_ENTRY || existing_key == hash_val) {
                bool collison_flag = false;
                while (true) {
                    if (existing_value < i) {
                        // occupied entry, but no need for swap, just check if
                        // collision
                        if (!tuple_eq(target->tuples[existing_value], cur_tuple,
                                      target->index_column_size)) {
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
                        if (!tuple_eq(target->tuples[existing_value], cur_tuple,
                                      target->index_column_size)) {
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
                        existing_value =
                            atomicCAS(&(target->index_map[position].value),
                                      existing_value, i);
                    }
                }
                if (!collison_flag) {
                    break;
                }
            }

            position = (position + 1) % target->index_map_size;
        }
    }
}

__global__ void count_index_entry_size(GHashRelContainer *target,
                                       tuple_size_t *size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->index_map_size)
        return;

    u64 stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < target->index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            atomicAdd(size, 1);
        }
    }
}

__global__ void shrink_index_map(GHashRelContainer *target,
                                 MEntity *old_index_map,
                                 tuple_size_t old_index_map_size) {
    tuple_size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= old_index_map_size)
        return;

    tuple_size_t stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < old_index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            u64 hash_val = target->index_map[i].key;
            tuple_size_t position = hash_val % target->index_map_size;
            while (true) {
                u64 existing_key = atomicCAS(&target->index_map[position].key,
                                             EMPTY_HASH_ENTRY, hash_val);
                if (existing_key == EMPTY_HASH_ENTRY) {
                    target->index_map[position].key = hash_val;
                    break;
                } else if (existing_key == hash_val) {
                    // hash for tuple's index column has already been recorded
                    break;
                }
                position = (position + 1) % target->index_map_size;
            }
        }
    }
}

__global__ void init_index_map(GHashRelContainer *target) {
    auto source = target->index_map;
    auto source_rows = target->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < source_rows; i += stride) {
        source[i].key = EMPTY_HASH_ENTRY;
        source[i].value = EMPTY_HASH_ENTRY;
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

__global__ void get_join_result_size2(GHashRelContainer *inner_table,
                                     GHashRelContainer *outer_table1,
                                     GHashRelContainer *outertable2,
                                     int join_column_counts,
                                     tuple_generator_hook tp_gen,
                                     tuple_predicate tp_pred,
                                     tuple_size_t *join_result_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table1->tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_table1->tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_table1->tuples[i];

        tuple_size_t current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table1->index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_table->index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val &&
                tuple_eq(
                    outer_tuple,
                    inner_table
                        ->tuples[inner_table->index_map[index_position].value],
                    outer_table1->index_column_size)) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (index_not_exists) {
            continue;
        }
        // pull all joined elements
        tuple_size_t position = inner_table->index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_table->tuples[position];
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
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
            if (position > inner_table->tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
    }
}

__global__ void get_join_result_size(GHashRelContainer *inner_table,
                                     GHashRelContainer *outer_table,
                                     int join_column_counts,
                                     tuple_generator_hook tp_gen,
                                     tuple_predicate tp_pred,
                                     tuple_size_t *join_result_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_table->tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_table->tuples[i];

        tuple_size_t current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_table->index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val &&
                tuple_eq(
                    outer_tuple,
                    inner_table
                        ->tuples[inner_table->index_map[index_position].value],
                    outer_table->index_column_size)) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (index_not_exists) {
            continue;
        }
        // pull all joined elements
        tuple_size_t position = inner_table->index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_table->tuples[position];
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
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
            if (position > inner_table->tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
    }
}

__global__ void
get_join_result(GHashRelContainer *inner_table, GHashRelContainer *outer_table,
                int join_column_counts, tuple_generator_hook tp_gen,
                tuple_predicate tp_pred, int output_arity,
                column_type *output_raw_data, tuple_size_t *res_count_array,
                tuple_size_t *res_offset, JoinDirection direction) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_table->tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        tuple_type outer_tuple = outer_table->tuples[i];

        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_table->index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val &&
                tuple_eq(
                    outer_tuple,
                    inner_table
                        ->tuples[inner_table->index_map[index_position].value],
                    outer_table->index_column_size)) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (index_not_exists) {
            continue;
        }

        // pull all joined elements
        tuple_size_t position = inner_table->index_map[index_position].value;
        while (true) {
            // TODO: always put join columns ahead? could be various benefits
            // but memory is issue to mantain multiple copies
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // tuple prefix match, join here
                tuple_type inner_tuple = inner_table->tuples[position];
                tuple_type new_tuple =
                    output_raw_data +
                    (res_offset[i] + current_new_tuple_cnt) * output_arity;

                // for (int j = 0; j < output_arity; j++) {
                // TODO: this will cause max arity of a relation is 20
                if (tp_gen != nullptr && tp_pred != nullptr) {
                    column_type tmp[20];
                    (*tp_gen)(inner_tuple, outer_tuple, tmp);
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
            if (position > (inner_table->tuple_counts - 1)) {
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
    tuple_type *tuple_full_buf;
    tuple_size_t new_full_size = current_full_size + delta->tuple_counts;

    bool extened_mem = false;

    tuple_size_t total_mem_size = get_total_memory();
    tuple_size_t free_mem = get_free_memory();
    u64 delta_mem_size = delta->tuple_counts * sizeof(tuple_type);
    int multiplier = FULL_BUFFER_VEC_MULTIPLIER;
    if (!pre_allocated_merge_buffer_flag && !fully_disable_merge_buffer_flag &&
        delta_mem_size * multiplier <= 0.1 * free_mem) {
        std::cout << "reenable pre-allocated merge buffer" << std::endl;
        pre_allocated_merge_buffer_flag = true;
    }

    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        if (tuple_merge_buffer_size <= new_full_size) {
            if (tuple_merge_buffer != nullptr) {
                checkCuda(cudaFree(tuple_merge_buffer));
                tuple_merge_buffer = nullptr;
            }
            std::cout << "extend mem" << std::endl;
            extened_mem = true;
            tuple_merge_buffer_size =
                current_full_size + (delta->tuple_counts * multiplier);
            u64 tuple_full_buf_mem_size =
                tuple_merge_buffer_size * sizeof(tuple_type);

            while (((free_mem - tuple_full_buf_mem_size) * 1.0 /
                    total_mem_size) < 0.4 &&
                   delta_mem_size * multiplier > 0.1 * free_mem) {
                std::cout << "multiplier : " << multiplier << std::endl;
                multiplier--;
                tuple_merge_buffer_size =
                    current_full_size + (delta->tuple_counts * multiplier);
                tuple_full_buf_mem_size =
                    tuple_merge_buffer_size * sizeof(tuple_type);
                if (multiplier == 2) {
                    std::cout << "not enough memory for merge buffer"
                              << std::endl;
                    // not enough space for pre-allocated buffer
                    pre_allocated_merge_buffer_flag = false;
                    tuple_merge_buffer_size = 0;
                    // cudaFree(tuple_merge_buffer);
                    checkCuda(cudaMalloc((void **)&tuple_full_buf,
                                         tuple_full_buf_mem_size));
                    break;
                }
            }
            if (pre_allocated_merge_buffer_flag) {
                checkCuda(cudaMalloc((void **)&tuple_merge_buffer,
                                     tuple_full_buf_mem_size));
                tuple_full_buf = tuple_merge_buffer;
            }
        } else {
            tuple_full_buf = tuple_merge_buffer;
        }
    } else {
        tuple_merge_buffer_size = current_full_size + delta->tuple_counts;
        u64 tuple_full_buf_mem_size =
            tuple_merge_buffer_size * sizeof(tuple_type);
        checkCuda(
            cudaMalloc((void **)&tuple_full_buf, tuple_full_buf_mem_size));
        // checkCuda(cudaMemset(tuple_full_buf, 0, tuple_full_buf_mem_size));
        // checkCuda(cudaDeviceSynchronize());
    }
    // std::cout << new_full_size << std::endl;

    timer.stop_timer();
    // std::cout << "malloc time : " << timer.get_spent_time() << std::endl;
    detail_time[0] = timer.get_spent_time();

    timer.start_timer();
    tuple_type *end_tuple_full_buf = thrust::merge(
        thrust::device, tuple_full, tuple_full + current_full_size,
        delta->tuples, delta->tuples + delta->tuple_counts, tuple_full_buf,
        tuple_indexed_less(delta->index_column_size, delta->arity));
    timer.stop_timer();
    // std::cout << "merge time : " << timer.get_spent_time() << std::endl;
    detail_time[1] = timer.get_spent_time();
    // checkCuda(cudaDeviceSynchronize());
    current_full_size = new_full_size;

    timer.start_timer();
    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        auto old_full = tuple_full;
        tuple_full = tuple_merge_buffer;
        tuple_merge_buffer = old_full;
        if (extened_mem) {
            checkCuda(cudaFree(tuple_merge_buffer));
            tuple_merge_buffer = nullptr;
            u64 tuple_full_buf_mem_size =
                tuple_merge_buffer_size * sizeof(tuple_type);
            checkCuda(cudaMalloc((void **)&tuple_merge_buffer,
                                 tuple_full_buf_mem_size));
        }
    } else {
        checkCuda(cudaFree(tuple_full));
        tuple_full = tuple_full_buf;
    }
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
    buffered_delta_vectors.push_back(delta);
    full->tuples = tuple_full;
    full->tuple_counts = current_full_size;
    if (index_flag) {
        reload_full_temp(full, arity, tuple_full, current_full_size,
                         index_column_size, dependent_column_size,
                         full->index_map_load_factor, grid_size, block_size);
    }
}

void load_relation_container(GHashRelContainer *target, int arity,
                             column_type *data, tuple_size_t data_row_size,
                             tuple_size_t index_column_size,
                             int dependent_column_size,
                             float index_map_load_factor, int grid_size,
                             int block_size, float *detail_time,
                             bool gpu_data_flag, bool sorted_flag,
                             bool build_index_flag, bool tuples_array_flag) {
    KernelTimer timer;
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    // load index selection into gpu
    // u64 index_columns_mem_size = index_column_size * sizeof(u64);
    // checkCuda(cudaMalloc((void**) &(target->index_columns),
    // index_columns_mem_size)); cudaMemcpy(target->index_columns,
    // index_columns, index_columns_mem_size, cudaMemcpyHostToDevice);
    if (data_row_size == 0) {
        return;
    }
    // load raw data from host
    if (gpu_data_flag) {
        target->data_raw = data;
    } else {
        u64 relation_mem_size =
            data_row_size * ((u64)arity) * sizeof(column_type);
        checkCuda(cudaMalloc((void **)&(target->data_raw), relation_mem_size));
        checkCuda(cudaMemcpy(target->data_raw, data, relation_mem_size,
                             cudaMemcpyHostToDevice));
    }
    if (tuples_array_flag) {
        // init tuple to be unsorted raw tuple data address
        u64 target_tuples_mem_size = data_row_size * sizeof(tuple_type);
        checkCuda(cudaMalloc((void **)&target->tuples, target_tuples_mem_size));
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemset(target->tuples, 0, target_tuples_mem_size));
        // std::cout << "grid size : " << grid_size << "    " << block_size <<
        // std::endl;
        init_tuples_unsorted<<<grid_size, block_size>>>(
            target->tuples, target->data_raw, arity, data_row_size);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    // sort raw data
    if (!sorted_flag) {
        timer.start_timer();
        if (arity <= RADIX_SORT_THRESHOLD) {
            sort_tuples(target->tuples, data_row_size, arity, index_column_size,
                        grid_size, block_size);
        } else {
            thrust::sort(thrust::device, target->tuples,
                         target->tuples + data_row_size,
                         tuple_indexed_less(index_column_size, arity));
            // checkCuda(cudaDeviceSynchronize());
        }
        // print_tuple_rows(target, "after sort");
        timer.stop_timer();
        detail_time[0] = timer.get_spent_time();
        // deduplication here?
        timer.start_timer();
        tuple_type *new_end =
            thrust::unique(thrust::device, target->tuples,
                           target->tuples + data_row_size, t_equal(arity));
        // checkCuda(cudaDeviceSynchronize());
        data_row_size = new_end - target->tuples;
        timer.stop_timer();
        detail_time[1] = timer.get_spent_time();
    }

    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    if (build_index_flag) {
        timer.start_timer();
        // init the index map
        // set the size of index map same as data, (this should give us almost
        // no conflict) however this can be memory inefficient
        target->index_map_size =
            std::ceil(data_row_size / index_map_load_factor);
        // target->index_map_size = data_row_size;
        u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
        checkCuda(
            cudaMalloc((void **)&(target->index_map), index_map_mem_size));
        checkCuda(cudaMemset(target->index_map, 0, index_map_mem_size));

        // load inited data struct into GPU memory
        GHashRelContainer *target_device;
        checkCuda(
            cudaMalloc((void **)&target_device, sizeof(GHashRelContainer)));
        checkCuda(cudaMemcpy(target_device, target, sizeof(GHashRelContainer),
                             cudaMemcpyHostToDevice));
        init_index_map<<<grid_size, block_size>>>(target_device);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        // std::cout << "finish init index map" << std::endl;
        // print_hashes(target, "after construct index map");
        // calculate hash
        calculate_index_hash<<<grid_size, block_size>>>(
            target_device,
            tuple_indexed_less(target->index_column_size, target->arity));
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaFree(target_device));
        timer.stop_timer();
        detail_time[2] = timer.get_spent_time();
    }
}

void repartition_relation_index(GHashRelContainer *target, int arity,
                                column_type *data, tuple_size_t data_row_size,
                                tuple_size_t index_column_size,
                                int dependent_column_size,
                                float index_map_load_factor, int grid_size,
                                int block_size, float *detail_time) {
    KernelTimer timer;
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    // load index selection into gpu
    // u64 index_columns_mem_size = index_column_size * sizeof(u64);
    // checkCuda(cudaMalloc((void**) &(target->index_columns),
    // index_columns_mem_size)); cudaMemcpy(target->index_columns,
    // index_columns, index_columns_mem_size, cudaMemcpyHostToDevice);
    if (data_row_size == 0) {
        return;
    }
    // load raw data from host
    target->data_raw = data;
    // init tuple to be unsorted raw tuple data address
    u64 target_tuples_mem_size = data_row_size * sizeof(tuple_type);
    checkCuda(cudaMalloc((void **)&target->tuples, target_tuples_mem_size));
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemset(target->tuples, 0, target_tuples_mem_size));
    // std::cout << "grid size : " << grid_size << "    " << block_size <<
    // std::endl;
    init_tuples_unsorted<<<grid_size, block_size>>>(
        target->tuples, target->data_raw, arity, data_row_size);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    timer.start_timer();
    if (arity <= RADIX_SORT_THRESHOLD) {
        sort_tuples(target->tuples, data_row_size, index_column_size,
                    index_column_size, grid_size, block_size);
    } else {
        thrust::sort(thrust::device, target->tuples,
                     target->tuples + data_row_size,
                     tuple_indexed_less(index_column_size, arity));

        checkCuda(cudaDeviceSynchronize());
    }
    // print_tuple_rows(target, "after sort");
    timer.stop_timer();
    detail_time[0] = timer.get_spent_time();
    detail_time[1] = timer.get_spent_time();
    if (arity <= RADIX_SORT_THRESHOLD) {
        sort_tuple_by_hash(target->tuples, data_row_size, arity,
                           index_column_size, grid_size, block_size);
    }

    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    timer.start_timer();
    // init the index map
    // set the size of index map same as data, (this should give us almost
    // no conflict) however this can be memory inefficient
    target->index_map_size = std::ceil(data_row_size / index_map_load_factor);
    // target->index_map_size = data_row_size;
    u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
    checkCuda(cudaMalloc((void **)&(target->index_map), index_map_mem_size));
    checkCuda(cudaMemset(target->index_map, 0, index_map_mem_size));

    // load inited data struct into GPU memory
    GHashRelContainer *target_device;
    checkCuda(cudaMalloc((void **)&target_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(target_device, target, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));
    init_index_map<<<grid_size, block_size>>>(target_device);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // std::cout << "finish init index map" << std::endl;
    // print_hashes(target, "after construct index map");
    // calculate hash
    calculate_index_hash<<<grid_size, block_size>>>(
        target_device,
        tuple_indexed_less(target->index_column_size, target->arity));
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(target_device));
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
}

void reload_full_temp(GHashRelContainer *target, int arity, tuple_type *tuples,
                      tuple_size_t data_row_size,
                      tuple_size_t index_column_size, int dependent_column_size,
                      float index_map_load_factor, int grid_size,
                      int block_size) {
    //
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    target->tuples = tuples;
    target->index_map_size = std::ceil(data_row_size / index_map_load_factor);
    // target->index_map_size = data_row_size;
    if (target->index_map != nullptr) {
        cudaFree(target->index_map);
    }
    u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
    checkCuda(cudaMalloc((void **)&(target->index_map), index_map_mem_size));
    cudaMemset(target->index_map, 0, index_map_mem_size);

    // load inited data struct into GPU memory
    GHashRelContainer *target_device;
    checkCuda(cudaMalloc((void **)&target_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(target_device, target, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
    init_index_map<<<grid_size, block_size>>>(target_device);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // std::cout << "finish init index map" << std::endl;
    // print_hashes(target, "after construct index map");
    // calculate hash
    calculate_index_hash<<<grid_size, block_size>>>(
        target_device,
        tuple_indexed_less(target->index_column_size, target->arity));
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(target_device));
}

void copy_relation_container(GHashRelContainer *dst, GHashRelContainer *src,
                             int grid_size, int block_size) {
    // dst->index_map_size = src->index_map_size;
    // dst->index_map_load_factor = src->index_map_load_factor;
    // checkCuda(cudaMalloc((void **)&dst->index_map,
    //                      dst->index_map_size * sizeof(MEntity)));
    // cudaMemcpy(dst->index_map, src->index_map,
    //            dst->index_map_size * sizeof(MEntity),
    //            cudaMemcpyDeviceToDevice);
    // dst->index_column_size = src->index_column_size;

    // dst->tuple_counts = src->tuple_counts;
    // dst->data_raw_row_size = src->data_raw_row_size;
    // dst->arity = src->arity;
    // checkCuda(cudaMalloc((void **)&dst->tuples,
    //                      dst->tuple_counts * sizeof(tuple_type)));
    // cudaMemcpy(dst->tuples, src->tuples, src->tuple_counts *
    // sizeof(tuple_type),
    //            cudaMemcpyDeviceToDevice);

    free_relation_container(dst);
    checkCuda(cudaMalloc((void **)&dst->data_raw,
                         src->arity * src->tuple_counts * sizeof(column_type)));
    checkCuda(cudaMemcpy(dst->data_raw, src->data_raw,
                         src->arity * src->tuple_counts * sizeof(column_type),
                         cudaMemcpyDeviceToDevice));
    float detail_time[5];
    load_relation_container(dst, src->arity, dst->data_raw, src->tuple_counts,
                            src->index_column_size, src->dependent_column_size,
                            0.8, grid_size, block_size, detail_time, true, true,
                            true);
}

void free_relation_container(GHashRelContainer *target) {
    target->tuple_counts = 0;
    target->index_map_size = 0;
    target->data_raw_row_size = 0;
    if (target->index_map != nullptr) {
        checkCuda(cudaFree(target->index_map));
        target->index_map = nullptr;
    }
    if (target->tuples != nullptr) {
        checkCuda(cudaFree(target->tuples));
        target->tuples = nullptr;
    }
    if (target->data_raw != nullptr) {
        checkCuda(cudaFree(target->data_raw));
        target->data_raw = nullptr;
    }
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
    load_relation_container(target->full, arity, data, data_row_size,
                            index_column_size, dependent_column_size, 0.8,
                            grid_size, block_size, detail_time);
}
