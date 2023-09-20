#pragma once
#include "relation.cuh"

/**
 * @brief binary join, close to local_join in slog's join RA operator
 *
 * @param inner
 * @param outer
 * @param block_size
 */
void binary_join(GHashRelContainer *inner, GHashRelContainer *outer,
                 GHashRelContainer *output_newt, int *reorder_array,
                 int reorder_array_size, JoinDirection direction, int grid_size,
                 int block_size, int iter, float *detail_time);
