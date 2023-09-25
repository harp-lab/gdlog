#include "../include/dynamic_dispatch.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/timer.cuh"
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include "../include/print.cuh"

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
        cudaMemcpy(rel->tuple_full, rel->full->tuples,
                   rel->full->tuple_counts * sizeof(tuple_type),
                   cudaMemcpyDeviceToDevice);
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
                cudaFree(rel->delta->index_map);
                rel->delta->index_map = nullptr;
            }
            if (rel->delta->tuples != nullptr) {
                cudaFree(rel->delta->tuples);
                rel->delta->tuples = nullptr;
            }

            timer.start_timer();
            if (rel->newt->tuple_counts == 0) {
                rel->delta =
                    new GHashRelContainer(rel->arity, rel->index_column_size,
                                          rel->dependent_column_size);
                std::cout << "iteration " << iteration_counter << " relation "
                          << rel->name << " no new tuple added" << std::endl;
                continue;
            }
            tuple_type *deduplicated_newt_tuples;
            checkCuda(cudaMalloc((void **)&deduplicated_newt_tuples,
                                 rel->newt->tuple_counts * sizeof(tuple_type)));
            //////

            tuple_type *deuplicated_end = thrust::set_difference(
                thrust::device, rel->newt->tuples,
                rel->newt->tuples + rel->newt->tuple_counts, rel->tuple_full,
                rel->tuple_full + rel->current_full_size,
                deduplicated_newt_tuples,
                tuple_indexed_less(rel->full->index_column_size,
                                   rel->full->arity -
                                       rel->dependent_column_size));
            checkCuda(cudaDeviceSynchronize());
            tuple_size_t deduplicate_size = deuplicated_end - deduplicated_newt_tuples;

            if (deduplicate_size != 0) {
                fixpoint_flag = false;
            }
            timer.stop_timer();
            set_diff_time += timer.get_spent_time();

            column_type *deduplicated_raw;
            checkCuda(cudaMalloc((void **)&deduplicated_raw,
                                 deduplicate_size * rel->newt->arity *
                                     sizeof(column_type)));
            flatten_tuples_raw_data<<<grid_size, block_size>>>(
                deduplicated_newt_tuples, deduplicated_raw, deduplicate_size,
                rel->newt->arity);
            checkCuda(cudaDeviceSynchronize());
            cudaFree(deduplicated_newt_tuples);

            free_relation_container(rel->newt);

            timer.start_timer();
            // cudaMallocHost((void **)&rel->delta, sizeof(GHashRelContainer));
            rel->delta = new GHashRelContainer(
                rel->arity, rel->index_column_size, rel->dependent_column_size);
            load_relation_container(rel->delta, rel->full->arity,
                                    deduplicated_raw, deduplicate_size,
                                    rel->full->index_column_size,
                                    rel->full->dependent_column_size,
                                    rel->full->index_map_load_factor, grid_size,
                                    block_size, true, true, true);
            checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            rebuild_delta_time += timer.get_spent_time();

            timer.start_timer();
            rel->flush_delta(grid_size, block_size);
            timer.stop_timer();
            merge_time += timer.get_spent_time();

            // print_tuple_rows(rel->full, "Path full after load newt");
            std::cout << "iteration " << iteration_counter << " relation "
                      << rel->name
                      << " finish dedup new tuples : " << deduplicate_size
                      << " delta tuple size: " << rel->delta->tuple_counts
                      << " full counts " << rel->current_full_size << std::endl;
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
        checkCuda(cudaMalloc((void **)&new_full_raw_data,
                             rel->current_full_size * rel->full->arity *
                                 sizeof(column_type)));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(
            rel->tuple_full, new_full_raw_data, rel->current_full_size,
            rel->full->arity);
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
