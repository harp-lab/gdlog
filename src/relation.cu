
#include "../include/exception.cuh"
#include "../include/relation.cuh"
#include "../include/print.cuh"
#include <thrust/sort.h>
#include <thrust/unique.h>

__global__ void calculate_index_hash(GHashRelContainer *target,
                                     tuple_indexed_less cmp) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->tuple_counts)
        return;

    u64 stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < target->tuple_counts; i += stride) {
        tuple_type cur_tuple = target->tuples[i];

        u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
        u64 request_hash_index = hash_val % target->index_map_size;
        u64 position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(target->index_map[position].key),
                                         EMPTY_HASH_ENTRY, hash_val);
            u64 existing_value = target->index_map[position].value;
            if (existing_key == EMPTY_HASH_ENTRY || existing_key == hash_val) {
                while (true) {
                    if (existing_value <= i) {
                        break;
                    } else {
                        // need swap
                        existing_value =
                            atomicCAS(&(target->index_map[position].value),
                                      existing_value, i);
                    }
                }
                break;
            }

            position = (position + 1) % target->index_map_size;
        }
    }
}

__global__ void count_index_entry_size(GHashRelContainer *target, u64 *size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->index_map_size)
        return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < target->index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            atomicAdd(size, 1);
        }
    }
}

__global__ void shrink_index_map(GHashRelContainer *target,
                                 MEntity *old_index_map,
                                 u64 old_index_map_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= old_index_map_size)
        return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < old_index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            u64 hash_val = target->index_map[i].key;
            u64 position = hash_val % target->index_map_size;
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

__global__ void acopy_entry(GHashRelContainer *source,
                            GHashRelContainer *destination) {
    auto source_rows = source->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        destination->index_map[i].key = source->index_map[i].key;
        destination->index_map[i].value = source->index_map[i].value;
    }
}

__global__ void acopy_data(GHashRelContainer *source,
                           GHashRelContainer *destination) {
    auto data_rows = source->tuple_counts;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= data_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < data_rows; i += stride) {
        tuple_type cur_src_tuple = source->tuples[i];
        for (int j = 0; j < source->arity; j++) {
            destination->data_raw[i * source->arity + j] = cur_src_tuple[j];
        }
        destination->tuples[i] = destination->tuples[i * source->arity];
    }
}

__global__ void init_index_map(GHashRelContainer *target) {
    auto source = target->index_map;
    auto source_rows = target->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < source_rows; i += stride) {
        source[i].key = EMPTY_HASH_ENTRY;
        source[i].value = EMPTY_HASH_ENTRY;
    }
}

__global__ void init_tuples_unsorted(tuple_type *tuples, column_type *raw_data,
                                     int arity, u64 rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < rows; i += stride) {
        tuples[i] = raw_data + i * arity;
    }
}

__global__ void get_join_result_size(GHashRelContainer *inner_table,
                                     GHashRelContainer *outer_table,
                                     int join_column_counts,
                                     u64 *join_result_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols,
        // outer_table->index_column_size * sizeof(column_type)); for (size_t
        // idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i *
        //     outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        u64 current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        u64 index_position = hash_val % inner_table->index_map_size;
        // 64 bit hash is less likely to have collision
        // partially solve hash conflict? maybe better for performance
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }
        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_table->tuples[position];
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
                                    join_column_counts);
            // if (outer_tuple[0] == 1966 && outer_tuple[1] == 8149  && iter ==
            // 1) {
            //     printf("init pos %lld, map_size: %lld, hash: %lld\n",
            //        index_position, inner_table->index_map_size,
            //        hash_val);
            //     printf("%d wwwwwwwwwwwwwwwwwwwwww %lld, %lld outer: %lld,
            //     %lld; inner: %lld, %lld;\n",
            //            cmp_res,
            //            inner_table->index_map[index_position].value,
            //            position,
            //            outer_tuple[0], outer_tuple[1],
            //            cur_inner_tuple[0], cur_inner_tuple[1]);
            // }
            if (cmp_res) {
                current_size++;
            } else {

                u64 inner_tuple_hash = prefix_hash(
                    cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
            }
            position = position + 1;
            if (position > inner_table->tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
        // cudaFree(outer_indexed_cols);
    }
}

__global__ void get_join_result(GHashRelContainer *inner_table,
                                GHashRelContainer *outer_table,
                                int join_column_counts,
                                tuple_generator_hook tp_gen, int output_arity,
                                column_type *output_raw_data,
                                u64 *res_count_array, u64 *res_offset,
                                JoinDirection direction) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols,
        // outer_table->index_column_size * sizeof(column_type)); for (size_t
        // idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i *
        //     outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        u64 index_position = hash_val % inner_table->index_map_size;
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }

        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
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

                for (int j = 0; j < output_arity; j++) {
                    (*tp_gen)(inner_tuple, outer_tuple, new_tuple);
                    // if (output_reorder_array[j] < inner_table->arity) {
                    //     new_tuple[j] = inner_tuple[output_reorder_array[j]];
                    // } else {
                    //     new_tuple[j] = outer_tuple[output_reorder_array[j] -
                    //                                inner_table->arity];
                    // }
                }
                current_new_tuple_cnt++;
                if (current_new_tuple_cnt > res_count_array[i]) {
                    break;
                }
            } else {
                // if not prefix not match, there might be hash collision
                tuple_type cur_inner_tuple = inner_table->tuples[position];
                u64 inner_tuple_hash = prefix_hash(
                    cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
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
                                        column_type *raw, u64 tuple_counts,
                                        int arity) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < tuple_counts; i += stride) {
        for (int j = 0; j < arity; j++) {
            raw[i * arity + j] = tuple_pointers[i][j];
        }
    }
}

void load_relation_container(GHashRelContainer *target, int arity,
                             column_type *data, u64 data_row_size,
                             u64 index_column_size, float index_map_load_factor,
                             int grid_size, int block_size, bool gpu_data_flag,
                             bool sorted_flag, bool build_index_flag,
                             bool tuples_array_flag) {
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
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
        cudaMemcpy(target->data_raw, data, relation_mem_size,
                   cudaMemcpyHostToDevice);
    }
    if (tuples_array_flag) {
        // init tuple to be unsorted raw tuple data address
        checkCuda(cudaMalloc((void **)&target->tuples,
                             data_row_size * sizeof(tuple_type)));
        init_tuples_unsorted<<<grid_size, block_size>>>(
            target->tuples, target->data_raw, arity, data_row_size);
    }
    // sort raw data
    if (!sorted_flag) {
        thrust::sort(thrust::device, target->tuples,
                     target->tuples + data_row_size,
                     tuple_indexed_less(index_column_size, arity));
        // print_tuple_rows(target, "after sort");

        // deduplication here?
        tuple_type *new_end =
            thrust::unique(thrust::device, target->tuples,
                           target->tuples + data_row_size, t_equal(arity));
        data_row_size = new_end - target->tuples;
    }

    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    if (build_index_flag) {
        // init the index map
        // set the size of index map same as data, (this should give us almost
        // no conflict) however this can be memory inefficient
        target->index_map_size =
            std::ceil(data_row_size / index_map_load_factor);
        // target->index_map_size = data_row_size;
        u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
        checkCuda(
            cudaMalloc((void **)&(target->index_map), index_map_mem_size));

        // load inited data struct into GPU memory
        GHashRelContainer *target_device;
        checkCuda(
            cudaMalloc((void **)&target_device, sizeof(GHashRelContainer)));
        cudaMemcpy(target_device, target, sizeof(GHashRelContainer),
                   cudaMemcpyHostToDevice);
        init_index_map<<<grid_size, block_size>>>(target_device);
        // std::cout << "finish init index map" << std::endl;
        // print_hashes(target, "after construct index map");
        // calculate hash
        calculate_index_hash<<<grid_size, block_size>>>(
            target_device,
            tuple_indexed_less(target->index_column_size, target->arity));
        cudaFree(target_device);
    }
}

void copy_relation_container(GHashRelContainer *dst, GHashRelContainer *src) {
    dst->index_map_size = src->index_map_size;
    dst->index_map_load_factor = src->index_map_load_factor;
    checkCuda(cudaMalloc((void **)&dst->index_map,
                         dst->index_map_size * sizeof(MEntity)));
    cudaMemcpy(dst->index_map, src->index_map,
               dst->index_map_size * sizeof(MEntity), cudaMemcpyDeviceToDevice);
    dst->index_column_size = src->index_column_size;

    dst->tuple_counts = src->tuple_counts;
    dst->data_raw_row_size = src->data_raw_row_size;
    dst->arity = src->arity;
    checkCuda(cudaMalloc((void **)&dst->tuples,
                         dst->tuple_counts * sizeof(tuple_type)));
    cudaMemcpy(dst->tuples, src->tuples, src->tuple_counts * sizeof(tuple_type),
               cudaMemcpyDeviceToDevice);
    checkCuda(
        cudaMalloc((void **)&dst->data_raw,
                   dst->arity * dst->data_raw_row_size * sizeof(column_type)));
    cudaMemcpy(dst->data_raw, src->data_raw,
               dst->arity * dst->data_raw_row_size * sizeof(column_type),
               cudaMemcpyDeviceToDevice);
}

void free_relation_container(GHashRelContainer *target) {
    target->tuple_counts = 0;
    if (target->index_map != nullptr)
        cudaFree(target->index_map);
    if (target->tuples != nullptr)
        cudaFree(target->tuples);
    if (target->data_raw != nullptr)
        cudaFree(target->data_raw);
}

void load_relation(Relation *target, std::string name, int arity,
                   column_type *data, u64 data_row_size, u64 index_column_size,
                   int grid_size, int block_size) {

    target->name = name;
    target->arity = arity;
    target->index_column_size = index_column_size;
    target->full = new GHashRelContainer(arity, index_column_size);
    target->delta = new GHashRelContainer(arity, index_column_size);
    target->newt = new GHashRelContainer(arity, index_column_size);

    load_relation_container(target->full, arity, data, data_row_size,
                            index_column_size, 0.8, grid_size, block_size);
}
