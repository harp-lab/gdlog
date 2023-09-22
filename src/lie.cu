#include "../include/dynamic_dispatch.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/timer.cuh"
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>

#include <variant>


void LIE::add_ra(ra_op op) {
    ra_ops.push_back(op);
}

void LIE::add_relations(Relation* rel, bool static_flag) {
    if (static_flag) {
        static_relations.push_back(rel);
    } else {
        update_relations.push_back(rel);
        // add delta and newt for it
    }
}

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
        copy_relation_container(rel->delta, rel->full);
    }

    while (true) {
        for (auto &ra_op : ra_ops) {
            timer.start_timer();
            std::visit(dynamic_dispatch{[](RelationalJoin &op) {
                                            // timer.start_timer();
                                            op();
                                        },
                                        [](RelationalCopy &op) {
                                            if (op.src_ver == FULL) {
                                                if (!op.copied) {
                                                    op();
                                                }
                                            } else {
                                                op();
                                            }
                                        }},
                       ra_op);
            timer.stop_timer();
            join_time += timer.get_spent_time();
        }

        // std::cout << "popluating new tuple" << std::endl;
        // merge delta into full
        bool fixpoint_flag = true;
        for (Relation *rel : update_relations) {

            // if (rel->newt->tuple_counts != 0) {  
            //     fixpoint_flag = false;
            // }
            timer.start_timer();
            if (iteration_counter != 0 && rel->delta->tuple_counts != 0) {
                // std::cout << rel->name << " merging ... " << rel->newt->tuple_counts << std::endl; 
                tuple_type *tuple_full_buf;
                checkCuda(cudaMalloc(
                    (void **)&tuple_full_buf,
                    (rel->current_full_size + rel->delta->tuple_counts) *
                        sizeof(tuple_type)));
                checkCuda(cudaDeviceSynchronize());
                tuple_type *end_tuple_full_buf = thrust::merge(
                    thrust::device, rel->tuple_full,
                    rel->tuple_full + rel->current_full_size,
                    rel->delta->tuples,
                    rel->delta->tuples + rel->delta->tuple_counts,
                    tuple_full_buf,
                    tuple_indexed_less(rel->delta->index_column_size,
                                       rel->delta->arity));
                checkCuda(cudaDeviceSynchronize());
                rel->current_full_size = end_tuple_full_buf - tuple_full_buf;
                cudaFree(rel->tuple_full);
                rel->tuple_full = tuple_full_buf;
            }
            rel->buffered_delta_vectors.push_back(rel->delta);
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
            timer.stop_timer();
            merge_time += timer.get_spent_time();

            timer.start_timer();
            if (rel->newt->tuple_counts == 0) {
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
                                   rel->full->arity));
            checkCuda(cudaDeviceSynchronize());
            u64 deduplicate_size = deuplicated_end - deduplicated_newt_tuples;

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
            cudaMallocHost((void **)&rel->delta, sizeof(GHashRelContainer));
            // rel->delta = new GHashRelContainer();
            load_relation_container(rel->delta, rel->full->arity,
                                    deduplicated_raw, deduplicate_size,
                                    rel->full->index_column_size,
                                    rel->full->index_map_load_factor, grid_size,
                                    block_size, true, true, true);
            timer.stop_timer();
            rebuild_delta_time += timer.get_spent_time();

            // print_tuple_rows(path_full, "Path full after load newt");
            std::cout << "iteration " << iteration_counter << " relation "
                      << rel->name
                      << " finish dedup new tuples : " << deduplicate_size
                      << " delta tuple size: " << rel->delta->tuple_counts
                      << " full counts " << rel->current_full_size << std::endl;
            iteration_counter++;
        }

        if (fixpoint_flag) {
            break;
        }
    }
    // merge full after reach fixpoint
    timer.start_timer();
    for (Relation *rel : update_relations) {
        if (rel->current_full_size <= rel->full->tuple_counts) {
            continue;
        }
        column_type *new_full_raw_data;
        checkCuda(cudaMalloc((void **)&new_full_raw_data,
                             rel->current_full_size * rel->full->arity *
                                 sizeof(column_type)));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(
            rel->tuple_full, new_full_raw_data, rel->current_full_size,
            rel->full->arity);
        checkCuda(cudaDeviceSynchronize());
        // cudaFree(tuple_merge_buffer);
        load_relation_container(rel->full, rel->full->arity, new_full_raw_data,
                                rel->current_full_size,
                                rel->full->index_column_size,
                                rel->full->index_map_load_factor, grid_size,
                                block_size, true, true);
        checkCuda(cudaDeviceSynchronize());
        std::cout << "Finished! path has " << rel->full->tuple_counts
                  << std::endl;
    }
    timer.stop_timer();
    float merge_full_time = timer.get_spent_time();

    std::cout << "Join time: " << join_time << " ; merge time: " << merge_time
              << " ; rebuild full time: " << merge_full_time
              << " ; rebuild delta time: " << rebuild_delta_time
              << " ; set diff time: " << set_diff_time << std::endl;
}
