#include "../include/exception.cuh"
#include "../include/print.cuh"
#include <iostream>
#include <rmm/device_vector.hpp>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void print_hashes(GHashRelContainer *target, const char *rel_name) {
    thrust::host_vector<MEntity> host_map = target->index_map;
    std::cout << "Relation hash >>> " << rel_name << std::endl;
    for (tuple_size_t i = 0; i < target->index_map_size; i++) {
        std::cout << host_map[i].key << "    " << host_map[i].value
                  << std::endl;
    }
    std::cout << "end <<<" << std::endl;
}

void print_tuple_rows(GHashRelContainer *target, const char *rel_name,
                      bool sort_flag) {
    // sort first
    rmm::device_vector<tuple_type> natural_ordered = target->tuples;
    if (sort_flag) {
        thrust::sort(thrust::device, natural_ordered.begin(),
                     natural_ordered.begin() + target->tuple_counts,
                     tuple_weak_less(target->arity));
    }
    // thrust::host_vector<tuple_type> tuples_host = natural_ordered;
    std::cout << "Relation tuples >>> " << rel_name << std::endl;
    std::cout << "Total tuples counts:  " << target->tuple_counts << std::endl;
    u32 pt_size = target->tuple_counts;
    // if (target->tuple_counts > 3000) {
    //     pt_size = 100;
    // }
    for (tuple_size_t i = 0; i < pt_size; i++) {
        tuple_type cur_tuple = natural_ordered[i];
        if (cur_tuple == nullptr) {
            std::cout << "null tuple" << std::endl;
            continue;
        }

        tuple_type cur_tuple_host;
        cudaMallocHost((void **)&cur_tuple_host,
                       target->arity * sizeof(column_type));
        cudaMemcpy(cur_tuple_host, cur_tuple,
                   target->arity * sizeof(column_type), cudaMemcpyDeviceToHost);
        // if (cur_tuple_host[0] != 1966) {
        //     continue;
        // }
        for (int j = 0; j < target->arity; j++) {
            std::cout << cur_tuple_host[j] << "\t";
        }
        std::cout << std::endl;
        cudaFreeHost(cur_tuple_host);
    }
    // if (target->tuple_counts > 3000) {
    //     std::cout << "........." << std::endl;
    // }
    std::cout << "end <<<" << std::endl;
}

void print_tuple_raw_data(GHashRelContainer *target, const char *rel_name) {
    std::cout << "Relation raw tuples >>> " << rel_name << std::endl;
    std::cout << "Total raw tuples counts:  " << target->data_raw_row_size
              << std::endl;
    column_type *cur_tuple_host;
    cudaMallocHost((void **)&cur_tuple_host,
                   target->arity * sizeof(column_type));
    for (tuple_size_t i = 0; i < target->data_raw_row_size; i++) {
        cudaMemcpy(cur_tuple_host,
                   target->data_raw.data().get() + i * target->arity,
                   target->arity * sizeof(column_type), cudaMemcpyDeviceToHost);
        for (int j = 0; j < target->arity; j++) {
            std::cout << cur_tuple_host[j] << "    ";
        }
        std::cout << std::endl;
    }
    cudaFreeHost(cur_tuple_host);
}

void print_memory_usage() {
    int num_gpus;
    size_t free, total;
    // cudaGetDeviceCount( &num_gpus );
    // for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
    // cudaSetDevice( gpu_id );
    // int id = 0;
    // cudaGetDevice( &id );
    cudaMemGetInfo(&free, &total);
    std::cout << "GPU " << 0 << " memory: free=" << free << ", total=" << total
              << std::endl;
    // }
}

tuple_size_t get_free_memory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

tuple_size_t get_total_memory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total;
}

// void print_tuple_list(tuple_type* tuples, tuple_size_t rows, tuple_size_t
// arity) {
//     tuple_type* tuples_host;
//     cudaMallocHost((void**) &tuples_host, rows * sizeof(tuple_type));
//     cudaMemcpy(tuples_host, tuples, rows * sizeof(tuple_type),
//                cudaMemcpyDeviceToHost);
//     if (rows > 100) {
//         rows = 100;
//     }
//     for (tuple_size_t i = 0; i < rows; i++) {
//         tuple_type cur_tuple = tuples_host[i];

//         tuple_type cur_tuple_host;
//         cudaMallocHost((void**) &cur_tuple_host, arity *
//         sizeof(column_type)); cudaMemcpy(cur_tuple_host, cur_tuple, arity *
//         sizeof(column_type),
//                    cudaMemcpyDeviceToHost);
//         for (tuple_size_t j = 0; j < arity; j++) {
//             std::cout << cur_tuple_host[j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }
