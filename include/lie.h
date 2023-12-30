#pragma once
#include "relational_algebra.h"
#include <climits>
#include <vector>

/**
 * @brief Logical inference engine(LIE). Compute fixpoint for a datalog rule SCC
 * (Strongly Connected Component).
 *
 */
struct LIE {
    // all relation operator used in this LIE
    std::vector<ra_op> ra_ops;

    // all relations may have new data in this SCC
    std::vector<Relation *> update_relations;
    // all relation won't be changed in this SCC
    std::vector<Relation *> static_relations;

    // temporary relations, these relations's FULL version won't be stored,
    // delta version of these relation will be cleared after used in join
    std::vector<Relation *> tmp_relations;

    // GPU grid size
    int grid_size;
    // GPU block size
    int block_size;

    bool reload_full_flag = true;
    int max_iteration = INT_MAX;

    LIE(int grid_size, int block_size)
        : grid_size(grid_size), block_size(block_size) {}

    /**
     * @brief compute fixpoint for current LIE
     *
     */
    void fixpoint_loop();

    /**
     * @brief Add a relation to SCC, all relation must be added before fixpoint
     * loop begin
     *
     * @param rel relation to add
     * @param static_flag whether a relation appears in output relation position
     * or not
     */
    void add_relations(Relation *rel, bool static_flag);

    /**
     * @brief add a temporary relation (a relation only have DELTA/NEWT)
     * 
     * @param rel 
     */
    void add_tmp_relation(Relation *rel);

    /**
     * @brief add a Relation Algebra operation
     * 
     * @param op 
     */
    void add_ra(ra_op op);
    // void ra(ra_op op);
};
