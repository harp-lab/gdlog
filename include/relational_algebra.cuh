#pragma once
#include "relation.cuh"
#include "tuple.cuh"
#include <thrust/host_vector.h>
#include <variant>

// function hook describ how inner and outer tuple are reordered to result tuple

struct RelationalJoin {

    Relation *inner_rel;
    RelationVersion inner_ver;
    Relation *outer_rel;
    RelationVersion outer_ver;

    Relation *output_rel;
    tuple_generator_hook tuple_generator;

    JoinDirection direction;
    int grid_size;
    int block_size;
    float *detail_time;

    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, tuple_generator_hook tp_gen,
                   JoinDirection direction, int grid_size, int block_size,
                   float *detail_time)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          direction(direction), grid_size(grid_size), block_size(block_size),
          detail_time(detail_time){};

    void operator()();
};

struct RelationalCopy {
    GHashRelContainer *src;
    GHashRelContainer *dest;

    tuple_generator_hook tuple_generator;
};

using ra_op = std::variant<RelationalJoin, RelationalCopy>;

enum RAtypes { JOIN, COPY };
// struct RelationalCopy {

// };

// /**
//  * @brief binary join, close to local_join in slog's join RA operator
//  *
//  * @param inner
//  * @param outer
//  * @param block_size
//  */
// void binary_join(GHashRelContainer *inner, GHashRelContainer *outer,
//                  GHashRelContainer *output_newt, tuple_generator_hook tp_gen,
//                  int reorder_array_size, JoinDirection direction, int
//                  grid_size, int block_size, int iter, float *detail_time);
