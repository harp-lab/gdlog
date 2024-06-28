#pragma once
#include "relation.h"
#include "tuple.h"
#include <thrust/host_vector.h>
#include <variant>

// for fixing
#ifndef MAX_REDUCE_SIZE
#define MAX_REDUCE_SIZE 80000000
#endif

// function hook describ how inner and outer tuple are reordered to result tuple

/**
 * @brief Relation Algerbra kernal for JOIN ⋈
 *
 */
struct RelationalJoin {

    // relation to compare, this relation must has index
    Relation *inner_rel;
    RelationVersion inner_ver;
    // serialized relation, every tuple in this relation will be iterated and
    // joined with tuples in inner relation
    Relation *outer_rel;
    RelationVersion outer_ver;

    // the relation to store the generated join result
    Relation *output_rel;
    // hook function will be mapped on every join result tuple
    tuple_generator_hook tuple_generator;
    // filter to be applied on every join result tuple
    tuple_predicate tuple_pred;

    // TODO: reserved for optimization
    JoinDirection direction;
    int grid_size;
    int block_size;

    // flag for benchmark, this will disable sorting on result
    bool disable_load = false;

    // join time for debug and profiling
    float *detail_time;

    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, tuple_generator_hook tp_gen,
                   tuple_predicate tp_pred, JoinDirection direction,
                   int grid_size, int block_size, float *detail_time)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          tuple_pred(tp_pred), direction(direction), grid_size(grid_size),
          block_size(block_size), detail_time(detail_time){};

    void operator()();
};

/**
 * @brief Relation Algerbra kernal for PROJECTION Π
 *
 */
struct RelationalCopy {
    Relation *src_rel;
    RelationVersion src_ver;
    Relation *dest_rel;
    tuple_copy_hook tuple_generator;
    tuple_predicate tuple_pred;

    int grid_size;
    int block_size;
    bool copied = false;

    RelationalCopy(Relation *src, RelationVersion src_ver, Relation *dest,
                   tuple_copy_hook tuple_generator, tuple_predicate tuple_pred,
                   int grid_size, int block_size)
        : src_rel(src), src_ver(src_ver), dest_rel(dest),
          tuple_generator(tuple_generator), tuple_pred(tuple_pred),
          grid_size(grid_size), block_size(block_size) {}

    void operator()();
};

/**
 * @brief Relation Algebra kernel for sync up different indices of the same
 * relation. This RA operator must be added in the end of each SCC, it will
 * directly change the DELTA version of dest relation
 *
 */
struct RelationalACopy {
    Relation *src_rel;
    Relation *dest_rel;
    // function will be mapped on all tuple copied
    tuple_copy_hook tuple_generator;
    // filter for copied tuple
    tuple_predicate tuple_pred;

    int grid_size;
    int block_size;

    RelationalACopy(Relation *src, Relation *dest,
                    tuple_copy_hook tuple_generator, tuple_predicate tuple_pred,
                    int grid_size, int block_size)
        : src_rel(src), dest_rel(dest), tuple_generator(tuple_generator),
          tuple_pred(tuple_pred), grid_size(grid_size), block_size(block_size) {
    }

    void operator()();
};

/**
 * @brief possible RA types
 * 
 */
using ra_op = std::variant<RelationalJoin, RelationalCopy, RelationalACopy>;

enum RAtypes { JOIN, COPY, ACOPY };
