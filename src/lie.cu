#include "../include/dynamic_dispatch.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"
// #include <future>
#include <iostream>
#include <map>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thread>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
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
    float memory_alloc_time = 0;
    float sort_time = 0;
    float unique_time = 0;
    float init_tp_array_time = 0;

    float join_get_size_time = 0;
    float join_get_result_time = 0;
    float rebuild_newt_time = 0;
    KernelTimer timer;

    float rebuild_rel_sort_time = 0;
    float rebuild_rel_unique_time = 0;
    float rebuild_rel_index_time = 0;

    std::cout << "start lie .... " << std::endl;
    // init full tuple buffer for all relation involved
    for (Relation *rel : update_relations) {
        copy_relation_container(rel->delta, rel->full, grid_size, block_size);
    }
    std::map<Relation *, std::thread> updated_res_map;

    rmm::device_vector<tuple_type> deduplicated_newt_buf;
    counting_buf_t join_counts_buf;
    counting_buf_t join_offset_buf;
    while (true) {
        for (auto &ra_op : ra_ops) {
            timer.start_timer();
            std::visit(
                dynamic_dispatch{
                    [&](RelationalJoin &op) {
                        // timer.start_timer();
                        if (updated_res_map.find(op.inner_rel) !=
                            updated_res_map.end()) {
                            // std::cout << "wait for "
                            //           << op.inner_rel->name
                            //           << std::endl;
                            if (updated_res_map[op.inner_rel].joinable()) {
                                updated_res_map[op.inner_rel].join();
                            }
                            updated_res_map.erase(op.inner_rel);
                        }
                        op(join_counts_buf, join_offset_buf);
                        join_counts_buf.clear();
                        join_offset_buf.clear();
                        // join_offset_buf.shrink_to_fit();
                        // join_counts_buf.shrink_to_fit();
                    },
                    [&](RelationalACopy &op) {
                        if (updated_res_map.find(op.src_rel) !=
                            updated_res_map.end()) {
                            if (updated_res_map[op.src_rel].joinable()) {
                                updated_res_map[op.src_rel].join();
                            }
                            updated_res_map.erase(op.src_rel);
                        }
                        op();
                    },
                    [&](RelationalCopy &op) {
                        if (updated_res_map.find(op.src_rel) !=
                            updated_res_map.end()) {
                            if (updated_res_map[op.src_rel].joinable()) {
                                updated_res_map[op.src_rel].join();
                            }
                            updated_res_map.erase(op.src_rel);
                        }
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

            // print_tuple_rows(rel->full, "Path full before populate
            // >>>>>>>>>>>>>>>> ", false);

            // populate newt
            timer.start_timer();
            rel->newt->tuples.resize(rel->newt->tuple_counts);
            init_tuples_unsorted<<<grid_size, block_size>>>(
                rel->newt->tuples.data().get(),
                rel->newt->data_raw.data().get(), rel->arity,
                rel->newt->tuple_counts);
            checkCuda(cudaStreamSynchronize(0));

            // print_tuple_raw_data(rel->newt, "Path newt before sort");

            timer.stop_timer();
            init_tp_array_time += timer.get_spent_time();
            timer.start_timer();
            thrust::sort(
                rmm::exec_policy(), rel->newt->tuples.begin(),
                rel->newt->tuples.end(),
                tuple_indexed_less(rel->index_column_size, rel->arity));
            timer.stop_timer();
            sort_time += timer.get_spent_time();
            timer.start_timer();
            auto new_end =
                thrust::unique(rmm::exec_policy(), rel->newt->tuples.begin(),
                               rel->newt->tuples.end(), t_equal(rel->arity));
            timer.stop_timer();
            rel->newt->tuple_counts = new_end - rel->newt->tuples.begin();
            rel->newt->tuples.resize(rel->newt->tuple_counts);
            unique_time += timer.get_spent_time();

            // if (rel->newt->tuple_counts != 0) {
            //     fixpoint_flag = false;
            // }
            if (iteration_counter == 0) {
                free_relation_container(rel->delta);
            }
            // drop the index of delta once merged, because it won't be used in
            // next iter when migrate more general case, this operation need to
            // be put off to end of all RA operation in current iteration
            rel->delta->index_map.clear();
            rel->delta->index_map.shrink_to_fit();
            rel->delta->tuples.clear();
            rel->delta->tuples.shrink_to_fit();
            // rel->delta->data_raw.shrink_to_fit();

            timer.start_timer();
            rel->delta = new GHashRelContainer(
                rel->arity, rel->index_column_size, rel->dependent_column_size);
            if (rel->newt->tuple_counts == 0) {
                std::cout << "iteration " << iteration_counter << " relation "
                          << rel->name << " no new tuple added" << std::endl;
                continue;
            }

            rel->delta->tuples.resize(rel->newt->tuple_counts);
            //////
            // print_tuple_rows(rel->newt, "Path newt before set diff", false);
            // print_tuple_rows(rel->full, "Path full before set diff
            // >>>>>>>>>>>>>>>> ", false);
            auto deuplicated_end = thrust::set_difference(
                rmm::exec_policy(), rel->newt->tuples.begin(),
                rel->newt->tuples.end(), rel->full->tuples.begin(),
                rel->full->tuples.begin() + rel->full->tuple_counts,
                rel->delta->tuples.begin(),
                tuple_indexed_less(rel->full->index_column_size,
                                   rel->full->arity -
                                       rel->dependent_column_size));
            // checkCuda(cudaStreamSynchronize(0));
            tuple_size_t deduplicate_size =
                deuplicated_end - rel->delta->tuples.begin();
            rel->delta->tuple_counts = deduplicate_size;
            if (deduplicate_size != 0) {
                fixpoint_flag = false;
            }
            timer.stop_timer();
            set_diff_time += timer.get_spent_time();
            if (updated_res_map.find(rel) != updated_res_map.end()) {
                if (updated_res_map[rel].joinable()) {
                    updated_res_map[rel].join();
                }
                updated_res_map.erase(rel);
            }
            timer.start_timer();
            rel->delta->data_raw.resize(deduplicate_size * rel->delta->arity);
            flatten_tuples_raw_data_thrust(rel->delta->tuples.data().get(),
                                           rel->delta->data_raw.data().get(),
                                           deduplicate_size, rel->delta->arity);
            rel->delta->data_raw.shrink_to_fit();
            // init_tuples_unsorted(tuple_type *tuples, column_type *raw_data,
            // int arity, tuple_size_t rows) use address in rel->delta->data_raw
            // to re init delta->tuples
            init_tuples_unsorted_thrust(rel->delta->tuples.data().get(),
                                        rel->delta->data_raw.data().get(),
                                        rel->delta->arity, deduplicate_size);
            timer.stop_timer();
            rebuild_delta_time += timer.get_spent_time();

            // TODO: do we need free it actually? we can just set counts to 0
            free_relation_container(rel->newt);
            rel->newt->tuple_counts = 0;
            rel->newt->data_raw_row_size = 0;

            timer.start_timer();
            rel->delta->update_index_map(grid_size, block_size);
            timer.stop_timer();

            rebuild_rel_index_time += timer.get_spent_time();

            // auto old_full = rel->tuple_full;
            float flush_detail_time[5] = {0, 0, 0, 0, 0};
            timer.start_timer();
            rel->flush_delta(grid_size, block_size, flush_detail_time);
            // updated_res_map[rel] = std::thread([&]() {
            //     rel->flush_delta(grid_size, block_size, flush_detail_time);
            // });
            timer.stop_timer();
            merge_time += flush_detail_time[1];
            memory_alloc_time += flush_detail_time[0];
            memory_alloc_time += flush_detail_time[2];
            // checkCuda(cudaFree(old_full));

            // print_tuple_rows(rel->full, "Path full after merge
            // >>>>>>>>>>>>>>>>>>>");
            std::cout << "iteration " << iteration_counter << " relation "
                      << rel->name
                      << " finish dedup new tuples : " << deduplicate_size
                      << " delta tuple size: " << rel->delta->tuple_counts
                      << " full counts " << rel->full->tuple_counts
                      << std::endl;
        }
        checkCuda(cudaStreamSynchronize(0));
        // std::cout << "Iteration " << iteration_counter << " finish
        // populating"
        //           << std::endl;
        print_memory_usage();
        std::cout << "Join time: " << join_time
                  << " ; sort newt time: " << sort_time
                  << " ; unique newt time: " << unique_time
                  << " ; merge full time: " << merge_time
                  << " ; memory alloc time: " << memory_alloc_time
                  << " ; rebuild delta time: " << rebuild_delta_time
                  << " ; set diff time: " << set_diff_time << std::endl;
        iteration_counter++;
        // if (iteration_counter >= 3) {
        //     break;
        // }

        if (fixpoint_flag || iteration_counter > max_iteration) {
            break;
        }
        // compute buffer/data_raw/tuples capcaity for each relation
        u_int64_t total_size = 0;
        u_int64_t total_size_full = 0;
        u_int64_t total_size_full_buf = 0;
        u_int64_t total_size_delta = 0;
        u_int64_t total_size_newt = 0;
        u_int64_t total_size_delta_size = 0;
        u_int64_t total_size_delta_map_size = 0;

        u_int64_t actual_size = 0;
        u_int64_t actual_size_full = 0;
        u_int64_t actual_size_full_buf = 0;
        u_int64_t actual_size_delta = 0;
        u_int64_t actual_size_newt = 0;

        for (Relation *rel : update_relations) {
            // full
            total_size_full +=
                rel->full->data_raw.capacity() * sizeof(column_type);
            total_size_full +=
                rel->full->tuples.capacity() * sizeof(tuple_type);
            total_size_full +=
                rel->full->index_map.capacity() * sizeof(MEntity);
            total_size_full_buf +=
                rel->tuple_merge_buffer.capacity() * sizeof(tuple_type);

            actual_size_full +=
                rel->full->data_raw.size() * sizeof(column_type);
            actual_size_full += rel->full->tuples.size() * sizeof(tuple_type);
            actual_size_full += rel->full->index_map.size() * sizeof(MEntity);
            actual_size_full_buf +=
                rel->tuple_merge_buffer.size() * sizeof(tuple_type);

            total_size_delta +=
                rel->delta->data_raw.capacity() * sizeof(column_type);
            total_size_delta +=
                rel->delta->tuples.capacity() * sizeof(tuple_type);
            total_size_delta +=
                rel->delta->index_map.capacity() * sizeof(MEntity);
            
            actual_size_delta +=
                rel->delta->data_raw.size() * sizeof(column_type);
            actual_size_delta += rel->delta->tuples.size() * sizeof(tuple_type);
            actual_size_delta +=
                rel->delta->index_map.size() * sizeof(MEntity);
            
            // delta
            for (auto &delta_b : rel->buffered_delta_vectors) {
                total_size_delta +=
                    delta_b->data_raw.capacity() * sizeof(column_type);
                total_size_delta +=
                    delta_b->tuples.capacity() * sizeof(tuple_type);
                total_size_delta +=
                    delta_b->index_map.capacity() * sizeof(MEntity);

                actual_size_delta +=
                    delta_b->data_raw.size() * sizeof(column_type);
                actual_size_delta +=
                    delta_b->tuples.size() * sizeof(tuple_type);
                actual_size_delta +=
                    delta_b->index_map.size() * sizeof(MEntity);

                total_size_delta_size += delta_b->tuples.size();
                total_size_delta_map_size += delta_b->index_map.size();
            }

            // newt
            total_size_newt +=
                rel->newt->data_raw.capacity() * sizeof(column_type);
            total_size_newt +=
                rel->newt->tuples.capacity() * sizeof(tuple_type);
            total_size_newt += rel->newt->index_map.capacity() *
                               sizeof(MEntity) * rel->newt->arity;

            actual_size_newt +=
                rel->newt->data_raw.size() * sizeof(column_type);
            actual_size_newt += rel->newt->tuples.size() * sizeof(tuple_type);
            actual_size_newt += rel->newt->index_map.size() * sizeof(MEntity) *
                                rel->newt->arity;
        }
        total_size += total_size_full + total_size_delta + total_size_newt +
                      total_size_full_buf;

        actual_size += actual_size_full + actual_size_delta + actual_size_newt +
                       actual_size_full_buf;
        std::cout << "Total size >>>>>>>>>>>>>>>" << std::endl;
        std::cout << "Iteration " << iteration_counter << " finish populating"
                  << " total size: " << total_size
                  << " full size: " << total_size_full
                  << " delta size: " << total_size_delta
                  << " newt size: " << total_size_newt
                  << " full buf size: " << total_size_full_buf
                //   << " delta size: " << total_size_delta_size
                //   << " delta map size: " << total_size_delta_map_size
                  << std::endl;
        std::cout << "Actual size >>>>>>>>>>>>>>>" << std::endl;
        std::cout << "Iteration " << iteration_counter << " finish populating"
                  << " actual size: " << actual_size
                  << " full size: " << actual_size_full
                  << " delta size: " << actual_size_delta
                  << " newt size: " << actual_size_newt
                  << " full buf size: " << actual_size_full_buf
                  << std::endl;
    }
    // 84942979072
    // 63177890936
    // 4717331424
    // 33436052608  delta
    // 22744183704  newt
    // 2280323200  full_buf

    print_memory_usage();

    for (auto &[rel, thread] : updated_res_map) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // merge full after reach fixpoint
    timer.start_timer();
    if (reload_full_flag) {
        std::cout << "Start merge full" << std::endl;
        for (Relation *rel : update_relations) {
            KernelTimer timer;
            // if (rel->current_full_size <= rel->full->tuple_counts) {
            //     continue;
            // }
            rel->full->data_raw.resize(rel->full->tuple_counts * rel->arity);
            // cudaFree(tuple_merge_buffer);
            timer.start_timer();
            flatten_tuples_raw_data_thrust(rel->full->tuples.data().get(),
                                           rel->full->data_raw.data().get(),
                                           rel->full->tuple_counts, rel->arity);
            timer.stop_timer();
            flatten_full_time += timer.get_spent_time();
            checkCuda(cudaStreamSynchronize(0));
            timer.start_timer();
            rel->full->update_index_map(grid_size, block_size);
            timer.stop_timer();
            rebuild_rel_index_time += timer.get_spent_time();
            std::cout << "Finished! " << rel->name << " has "
                      << rel->full->tuple_counts << std::endl;
            for (auto &delta_b : rel->buffered_delta_vectors) {
                free_relation_container(delta_b);
            }
            free_relation_container(rel->delta);
            free_relation_container(rel->newt);
        }
    } else {
        for (Relation *rel : update_relations) {
            std::cout << "Finished! " << rel->name << " has "
                      << rel->full->tuple_counts << std::endl;
        }
    }
    timer.stop_timer();
    float merge_full_time = timer.get_spent_time();

    cudaDeviceSynchronize();
    std::cout << "Join time: " << join_time
              << " ; merge full time: " << merge_time
              << " ; sort newt time: " << sort_time
              << " ; rebuild full time: " << merge_full_time
              << " ; rebuild delta time: " << rebuild_delta_time
              << " ; set diff time: " << set_diff_time << std::endl;
    std::cout << "Rebuild relation detail time : rebuild rel sort time: "
              << rebuild_rel_sort_time
              << " ; rebuild rel unique time: " << rebuild_rel_unique_time
              << " ; rebuild rel index time: " << rebuild_rel_index_time
              << std::endl;
}
