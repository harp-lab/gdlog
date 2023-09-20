#pragma once
#include "tuple.cuh"

/**
 * @brief A hash table entry
 * TODO: no need for struct actually, a u64[2] should be enough, easier to init
 *
 */
struct MEntity {
    // index position in actual index_arrary
    u64 key;
    // tuple position in actual data_arrary
    u64 value;
};

#define EMPTY_HASH_ENTRY ULLONG_MAX
/**
 * @brief a C-style hashset indexing based relation container.
 *        Actual data is still stored using sorted set.
 *        Different from normal btree relation, using hash table storing the
 * index to accelarte range fetch. Good:
 *           - fast range fetch, in Shovon's ATC paper it shows great
 * performance.
 *           - fast serialization, its very GPU friendly and also easier for MPI
 * inter-rank comm transmission. Bad:
 *           - need reconstruct index very time tuple is inserted (need more
 * reasonable algorithm).
 *           - sorting is a issue, each update need resort everything seems
 * stupid.
 *
 */
struct GHashRelContainer {
    // open addressing hashmap for indexing
    MEntity *index_map = nullptr;
    u64 index_map_size;
    float index_map_load_factor;

    // index prefix length
    // don't have to be u64,int is enough
    // u64 *index_columns;
    u64 index_column_size;

    // the pointer to flatten tuple, all tuple pointer here need to be sorted
    tuple_type *tuples = nullptr;
    // flatten tuple data
    column_type *data_raw = nullptr;
    // number of tuples
    u64 tuple_counts;
    // actual tuple rows in flatten data, this maybe different from
    // tuple_counts when deduplicated
    u64 data_raw_row_size;
    int arity;
};

enum JoinDirection { LEFT, RIGHT };

/**
 * @brief fill in index hash table for a relation in parallel, assume index is
 * correctly initialized, data has been loaded , deduplicated and sorted
 *
 * @param target the hashtable to init
 * @return dedeuplicated_bitmap
 */
__global__ void calculate_index_hash(GHashRelContainer *target,
                                     tuple_indexed_less cmp);

/**
 * @brief count how many non empty hash entry in index map
 *
 * @param target target relation hash table
 * @param size return the size
 * @return __global__
 */
__global__ void count_index_entry_size(GHashRelContainer *target, u64 *size);

/**
 * @brief rehash to make index map more compact, the new index hash size is
 * already update in target new index already inited to empty table and have new
 * size.
 *
 * @param target
 * @param old_index_map index map before compaction
 * @param old_index_map_size original size of index map before compaction
 * @return __global__
 */
__global__ void shrink_index_map(GHashRelContainer *target,
                                 MEntity *old_index_map,
                                 u64 old_index_map_size);

/**
 * @brief acopy the **index** from a relation to another, please use this
 * together with *copy_data*, and settle up all metadata before copy
 *
 * @param source source relation
 * @param destination destination relation
 * @return __global__
 */
__global__ void acopy_entry(GHashRelContainer *source,
                            GHashRelContainer *destination);
__global__ void acopy_data(GHashRelContainer *source,
                           GHashRelContainer *destination);

/**
 * @brief a CUDA kernel init the index entry map of a hashtabl
 *
 * @param target the hashtable to init
 * @return void
 */
__global__ void init_index_map(GHashRelContainer *target);

// a helper function to init an unsorted tuple arrary from raw data
__global__ void init_tuples_unsorted(tuple_type *tuples, column_type *raw_data,
                                     int arity, u64 rows);

/**
 * @brief for all tuples in outer table, match same prefix with inner table
 *
 * @note can we use pipeline here? since many matching may acually missing
 *
 * @param inner_table the hashtable to iterate
 * @param outer_table the hashtable to match
 * @param join_column_counts number of join columns (inner and outer must agree
 * on this)
 * @param  return value stored here, size of joined tuples
 * @return void
 */
__global__ void get_join_result_size(GHashRelContainer *inner_table,
                                     GHashRelContainer *outer_table,
                                     int join_column_counts,
                                     u64 *join_result_size, int iter,
                                     u64 *debug = nullptr);

/**
 * @brief compute the join result
 *
 * @param inner_table
 * @param outer_table
 * @param join_column_counts
 * @param output_reorder_array reorder array for output relation column
 * selection, arrary pos < inner->arity is index in inner, > is index in outer.
 * @param output_arity output relation arity
 * @param output_raw_data join result, need precompute the size
 * @return __global__
 */
__global__ void
get_join_result(GHashRelContainer *inner_table, GHashRelContainer *outer_table,
                int join_column_counts, int *output_reorder_array,
                int output_arity, column_type *output_raw_data,
                u64 *res_count_array, u64 *res_offset, JoinDirection direction);

__global__ void flatten_tuples_raw_data(tuple_type *tuple_pointers,
                                        column_type *raw, u64 tuple_counts,
                                        int arity);

//////////////////////////////////////////////////////
// CPU functions

/**
 * @brief load raw data into relation container
 *
 * @param target hashtable struct in host
 * @param arity
 * @param data raw data on host
 * @param data_row_size
 * @param index_columns index columns id in host
 * @param index_column_size
 * @param index_map_load_factor
 * @param grid_size
 * @param block_size
 * @param gpu_data_flag if data is a GPU memory address directly assign to
 * target's data_raw
 */
void load_relation(GHashRelContainer *target, int arity, column_type *data,
                   u64 data_row_size, u64 index_column_size,
                   float index_map_load_factor, int grid_size, int block_size,
                   bool gpu_data_flag = false, bool sorted_flag = false,
                   bool build_index_flag = true, bool tuples_array_flag = true);

/**
 * @brief copy a relation into an **empty** relation
 *
 * @param dst
 * @param src
 */
void copy_relation_container(GHashRelContainer *dst, GHashRelContainer *src);

/**
 * @brief clean all data in a relation container
 *
 * @param target
 */
void free_relation(GHashRelContainer *target);
