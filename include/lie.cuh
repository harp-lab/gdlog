#pragma once
#include "relational_algebra.cuh"
#include <vector>

struct LIE {
    std::vector<ra_op> ra_ops;

    std::vector<Relation *> update_relations;
    std::vector<Relation *> static_relations;
    std::vector<Relation *> tmp_relations;

    int grid_size;
    int block_size;

    LIE(int grid_size, int block_size)
        : grid_size(grid_size), block_size(block_size) {}

    void fixpoint_loop();
    void add_relations(Relation *rel, bool static_flag);
    void add_tmp_relation(Relation *rel);
    void add_ra(ra_op op);
    // void ra(ra_op op);
};
