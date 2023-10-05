#include "../include/dynamic_dispatch.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>

#include <variant>

void LIE::add_ra(ra_op op) { ra_ops.push_back(op); }

void LIE::add_relations(Relation *rel, bool static_flag) {
    if (static_flag) {
        static_relations.push_back(rel);
    } else {
        update_relations.push_back(rel);
        // add delta and newt for it
    }
}

void LIE::add_tmp_relation(Relation *rel) { tmp_relations.push_back(rel); }

void LIE::fixpoint_loop() {

    int iteration_counter = 0;
    float join_time = 0;
    float merge_time = 0;
    float rebuild_time = 0;
    float flatten_time = 0;
    float set_diff_time = 0;
    float rebuild_delta_time = 0;
    float flatten_full_time = 0;

    float join_get_size_time = 0;
    float join_get_result_time = 0;
    float rebuild_newt_time = 0;
    KernelTimer timer;

    std::cout << "start lie .... " << std::endl;
    // init full tuple buffer for all relation involved
    for (Relation *rel : update_relations) {
        checkCuda(cudaMalloc((void **)&rel->tuple_full,
                             rel->full->tuple_counts * sizeof(tuple_type)));
        checkCuda(cudaMemcpy(rel->tuple_full, rel->full->tuples,
                             rel->full->tuple_counts * sizeof(tuple_type),
                             cudaMemcpyDeviceToDevice));
        rel->current_full_size = rel->full->tuple_counts;
        copy_relation_container(rel->delta, rel->full, grid_size, block_size);
        checkCuda(cudaDeviceSynchronize());
        // std::cout << "wwwwwwwwww" << rel->delta->tuple_counts << std::endl;
    }

    while (true) {
        for (auto &ra_op : ra_ops) {
            timer.start_timer();
            std::visit(dynamic_dispatch{[](RelationalJoin &op) {
                                            // timer.start_timer();
                                            op();
                                        },
                                        [](RelationalACopy &op) { op(); },
                                        [](RelationalCopy &op) {
                                            if (op.src_ver == FULL) {
                                                if (!op.copied) {
                                                    op();
                                                    op.copied = true;
                                                }
                                            } else {
                                                op();
                                            }
                                        }},
                       ra_op);
            checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            join_time += timer.get_spent_time();
        }

        // clean tmp relation
        for (Relation *rel : tmp_relations) {
            free_relation_container(rel->newt);
        }

        std::cout << "Iteration " << iteration_counter
                  << " popluating new tuple" << std::endl;
        // merge delta into full
        bool fixpoint_flag = true;
        for (Relation *rel : update_relations) {
            std::cout << rel->name << std::endl;
            // if (rel->newt->tuple_counts != 0) {
            //     fixpoint_flag = false;
            // }
            if (iteration_counter == 0) {
                free_relation_container(rel->delta);
            }
            // drop the index of delta once merged, because it won't be used in
            // next iter when migrate more general case, this operation need to
            // be put off to end of all RA operation in current iteration
            if (rel->delta->index_map != nullptr) {
                checkCuda(cudaFree(rel->delta->index_map));
                rel->delta->index_map = nullptr;
            }
            if (rel->delta->tuples != nullptr) {
                checkCuda(cudaFree(rel->delta->tuples));
                rel->delta->tuples = nullptr;
            }

            if (rel->dep_pred == nullptr) {
            timer.start_timer();
            if (rel->newt->tuple_counts == 0) {
                rel->delta = new GHashRelContainer(
                    rel->arity, rel->index_column_size,
                    rel->dependent_column_size);
                std::cout << "iteration " << iteration_counter
                            << " relation " << rel->name
                            << " no new tuple added" << std::endl;
                continue;
            }
            tuple_type *deduplicated_newt_tuples;
            u64 deduplicated_newt_tuples_mem_size =
                rel->newt->tuple_counts * sizeof(tuple_type);
            checkCuda(cudaMalloc((void **)&deduplicated_newt_tuples,
                                    deduplicated_newt_tuples_mem_size));
            checkCuda(cudaMemset(deduplicated_newt_tuples, 0,
                                    deduplicated_newt_tuples_mem_size));

            tuple_type *deuplicated_end = thrust::set_difference(
                thrust::device, rel->newt->tuples,
                rel->newt->tuples + rel->newt->tuple_counts,
                rel->tuple_full, rel->tuple_full + rel->current_full_size,
                deduplicated_newt_tuples,
                tuple_indexed_less(rel->full->index_column_size,
                                    rel->full->arity -
                                        rel->dependent_column_size));
            checkCuda(cudaDeviceSynchronize());
            tuple_size_t deduplicate_size =
                deuplicated_end - deduplicated_newt_tuples;

            if (deduplicate_size != 0) {
                fixpoint_flag = false;
            }
            timer.stop_timer();
            set_diff_time += timer.get_spent_time();

            column_type *deduplicated_raw;
            u64 dedeuplicated_raw_mem_size =
                deduplicate_size * rel->newt->arity * sizeof(column_type);
            checkCuda(cudaMalloc((void **)&deduplicated_raw,
                                    dedeuplicated_raw_mem_size));
            checkCuda(cudaMemset(deduplicated_raw, 0,
                                    dedeuplicated_raw_mem_size));
            flatten_tuples_raw_data<<<grid_size, block_size>>>(
                deduplicated_newt_tuples, deduplicated_raw,
                deduplicate_size, rel->newt->arity);
            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
            checkCuda(cudaFree(deduplicated_newt_tuples));

            free_relation_container(rel->newt);

            timer.start_timer();
            rel->delta =
                new GHashRelContainer(rel->arity, rel->index_column_size,
                                        rel->dependent_column_size);
            load_relation_container(
                rel->delta, rel->full->arity, deduplicated_raw,
                deduplicate_size, rel->full->index_column_size,
                rel->full->dependent_column_size,
                rel->full->index_map_load_factor, grid_size, block_size,
                true, true, true);
            checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            rebuild_delta_time += timer.get_spent_time();

            timer.start_timer();
            rel->flush_delta(grid_size, block_size);
            timer.stop_timer();
            merge_time += timer.get_spent_time();

            // print_tuple_rows(rel->full, "Path full after load newt");
            // std::cout << "iteration " << iteration_counter << " relation
            // "
            //         << rel->name
            //         << " finish dedup new tuples : " << deduplicate_size
            //         << " delta tuple size: " << rel->delta->tuple_counts
            //         << " full counts " << rel->current_full_size <<
            //         std::endl;
            } else {
                // recursive aggregation
                // merge newt to full directly
                tuple_size_t new_full_size =
                    rel->current_full_size + rel->newt->tuple_counts;
                // std::cout << new_full_size << std::endl;
                tuple_type *tuple_full_buf;
                u64 tuple_full_buf_mem_size =
                    new_full_size * sizeof(tuple_type);
                checkCuda(cudaMalloc((void **)&tuple_full_buf,
                                     tuple_full_buf_mem_size));
                checkCuda(
                    cudaMemset(tuple_full_buf, 0, tuple_full_buf_mem_size));
                checkCuda(cudaDeviceSynchronize());

                tuple_type *end_tuple_full_buf = thrust::merge(
                    thrust::device, rel->tuple_full,
                    rel->tuple_full + rel->current_full_size, rel->newt->tuples,
                    rel->newt->tuples + rel->newt->tuple_counts, tuple_full_buf,
                    tuple_indexed_less(rel->full->index_column_size,
                                       rel->full->arity -
                                           rel->dependent_column_size,
                                       rel->dep_pred));
                checkCuda(cudaDeviceSynchronize());
                // after merge all tuple need aggregation has been gatered
                // together a full aggregation will be reduce them, but we need
                // monotonicity, so we use in place unique operation, which
                // whill keep the first occurence (smallest/largest) of
                // aggregated tuples sharing the same non-dependent columns
                tuple_type *deduplicated_tuple_full_buf_end = thrust::unique(
                    thrust::device, tuple_full_buf, end_tuple_full_buf,
                    t_equal(rel->full->arity - rel->dependent_column_size));
                tuple_size_t deduplicated_tuple_full_buf_size =
                    deduplicated_tuple_full_buf_end - tuple_full_buf;
                // then propagate the delta by set difference new and old full
                tuple_type *propogated_delta_tuples;
                checkCuda(
                    cudaMalloc((void **)&propogated_delta_tuples,
                               rel->newt->tuple_counts * sizeof(tuple_type)));
                tuple_type *propogated_delta_tuples_end =
                    thrust::set_difference(
                        thrust::device, tuple_full_buf,
                        deduplicated_tuple_full_buf_end, rel->tuple_full,
                        rel->tuple_full + rel->current_full_size,
                        propogated_delta_tuples,
                        tuple_indexed_less(rel->full->index_column_size,
                                           rel->full->arity));
                column_type *propogated_delta_raw;
                tuple_size_t propogated_delta_size =
                    propogated_delta_tuples_end - propogated_delta_tuples;
                u64 propogated_delta_raw_mem_size = propogated_delta_size *
                                                    rel->full->arity *
                                                    sizeof(column_type);
                checkCuda(cudaMalloc((void **)&propogated_delta_raw,
                                     propogated_delta_raw_mem_size));
                flatten_tuples_raw_data<<<grid_size, block_size>>>(
                    propogated_delta_tuples, propogated_delta_raw,
                    propogated_delta_size, rel->full->arity);
                checkCuda(cudaGetLastError());
                checkCuda(cudaDeviceSynchronize());
                checkCuda(cudaFree(propogated_delta_tuples));
                rel->delta =
                    new GHashRelContainer(rel->arity, rel->index_column_size,
                                          rel->dependent_column_size);
                load_relation_container(
                    rel->delta, rel->full->arity, propogated_delta_raw,
                    propogated_delta_size, rel->full->index_column_size,
                    rel->full->dependent_column_size,
                    rel->full->index_map_load_factor, grid_size, block_size,
                    true, true, true);
                rel->buffered_delta_vectors.push_back(rel->delta);

                // reload full, since merge will cause tuple inside newt
                // inserted into full if don't reload full, can't free newt
                // this operation need huge buffer for new full
                rel->current_full_size = deduplicated_tuple_full_buf_size;
                checkCuda(cudaFree(rel->tuple_full));
                column_type *new_full_raw_data;
                u64 new_full_raw_data_mem_size = rel->current_full_size *
                                                 rel->full->arity *
                                                 sizeof(column_type);
                checkCuda(cudaMalloc((void **)&new_full_raw_data,
                                     new_full_raw_data_mem_size));
                checkCuda(cudaMemset(new_full_raw_data, 0,
                                     new_full_raw_data_mem_size));
                flatten_tuples_raw_data<<<grid_size, block_size>>>(
                    rel->tuple_full, new_full_raw_data, rel->current_full_size,
                    rel->full->arity);
                checkCuda(cudaGetLastError());
                checkCuda(cudaDeviceSynchronize());
                free_relation_container(rel->newt);
                load_relation_container(
                    rel->full, rel->full->arity, new_full_raw_data,
                    rel->current_full_size, rel->full->index_column_size,
                    rel->full->dependent_column_size,
                    rel->full->index_map_load_factor, grid_size, block_size,
                    true, true, true);
                rel->tuple_full = rel->full->tuples;
                rel->current_full_size = rel->full->tuple_counts;
            }
        }
        checkCuda(cudaDeviceSynchronize());
        std::cout << "Iteration " << iteration_counter << " finish populating"
                  << std::endl;
        print_memory_usage();
        iteration_counter++;
        // if (iteration_counter >= 3) {
        //     break;
        // }

        if (fixpoint_flag) {
            break;
        }
    }
    // merge full after reach fixpoint
    timer.start_timer();
    for (Relation *rel : update_relations) {
        // if (rel->current_full_size <= rel->full->tuple_counts) {
        //     continue;
        // }
        column_type *new_full_raw_data;
        u64 new_full_raw_data_mem_size =
            rel->current_full_size * rel->full->arity * sizeof(column_type);
        checkCuda(cudaMalloc((void **)&new_full_raw_data,
                             new_full_raw_data_mem_size));
        checkCuda(cudaMemset(new_full_raw_data, 0, new_full_raw_data_mem_size));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(
            rel->tuple_full, new_full_raw_data, rel->current_full_size,
            rel->full->arity);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        // cudaFree(tuple_merge_buffer);
        load_relation_container(
            rel->full, rel->full->arity, new_full_raw_data,
            rel->current_full_size, rel->full->index_column_size,
            rel->full->dependent_column_size, rel->full->index_map_load_factor,
            grid_size, block_size, true, true, true);
        checkCuda(cudaDeviceSynchronize());
        std::cout << "Finished! " << rel->name << " has "
                  << rel->full->tuple_counts << std::endl;
        for (auto &delta_b : rel->buffered_delta_vectors) {
            free_relation_container(delta_b);
        }
        free_relation_container(rel->delta);
        free_relation_container(rel->newt);
    }
    timer.stop_timer();
    float merge_full_time = timer.get_spent_time();

    std::cout << "Join time: " << join_time
              << " ; merge full time: " << merge_time
              << " ; rebuild full time: " << merge_full_time
              << " ; rebuild delta time: " << rebuild_delta_time
              << " ; set diff time: " << set_diff_time << std::endl;
}
