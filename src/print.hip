#include "../include/print.h"
#include <iostream>
#include <thrust/sort.h>

void print_hashes(GHashRelContainer *target, const char *rel_name) {
    MEntity *host_map;
    hipMallocHost((void **)&host_map,
                   target->index_map_size * sizeof(MEntity));
    hipMemcpy(host_map, target->index_map,
               target->index_map_size * sizeof(MEntity),
               hipMemcpyDeviceToHost);
    std::cout << "Relation hash >>> " << rel_name << std::endl;
    for (tuple_size_t i = 0; i < target->index_map_size; i++) {
        std::cout << host_map[i].key << "    " << host_map[i].value
                  << std::endl;
    }
    std::cout << "end <<<" << std::endl;
    hipFreeHost(host_map);
}

void print_tuple_rows(GHashRelContainer* target, const char *rel_name) {
    // sort first
    tuple_type* natural_ordered;
    hipMalloc((void**) &natural_ordered, target->tuple_counts * sizeof(tuple_type));
    hipMemcpy(natural_ordered, target->tuples, target->tuple_counts * sizeof(tuple_type),
               hipMemcpyDeviceToDevice);
    thrust::sort(thrust::device, natural_ordered, natural_ordered+target->tuple_counts,
                 tuple_weak_less(target->arity));

    tuple_type* tuples_host;
    hipMallocHost((void**) &tuples_host, target->tuple_counts * sizeof(tuple_type));
    hipMemcpy(tuples_host, natural_ordered, target->tuple_counts * sizeof(tuple_type),
               hipMemcpyDeviceToHost);
    std::cout << "Relation tuples >>> " << rel_name << std::endl;
    std::cout << "Total tuples counts:  " <<  target->tuple_counts << std::endl;
    u32 pt_size = target->tuple_counts;
    if (target->tuple_counts > 3000) {
        pt_size = 3000;
    }
    for (tuple_size_t i = 0; i < pt_size; i++) {
        tuple_type cur_tuple = tuples_host[i];
        
        tuple_type cur_tuple_host;
        hipMallocHost((void**) &cur_tuple_host, target->arity * sizeof(column_type));
        hipMemcpy(cur_tuple_host, cur_tuple, target->arity * sizeof(column_type),
                   hipMemcpyDeviceToHost);
        // if (cur_tuple_host[0] != 1966) {
        //     continue;
        // }
        for (int j = 0; j < target->arity; j++) {

            std::cout << cur_tuple_host[j] << "\t";
        }
        std::cout << std::endl;
        hipFreeHost(cur_tuple_host);
    }
    if (target->tuple_counts > 3000) {
        std::cout << "........." << std::endl;
    }
    std::cout << "end <<<" << std::endl;

    hipFreeHost(tuples_host);
    hipFree(natural_ordered);
}

void print_tuple_raw_data(GHashRelContainer* target, const char *rel_name) {
    column_type* raw_data_host;
    u64 mem_raw = target->data_raw_row_size * target->arity * sizeof(column_type);
    hipMallocHost((void**) &raw_data_host, mem_raw);
    hipMemcpy(raw_data_host, target->data_raw, mem_raw, hipMemcpyDeviceToHost);
    std::cout << "Relation raw tuples >>> " << rel_name << std::endl;
    std::cout << "Total raw tuples counts:  " <<  target->data_raw_row_size << std::endl;
    for (tuple_size_t i = 0; i < target->data_raw_row_size; i++) {
        if (raw_data_host[i*target->arity] != 3) {
            continue;
        }
        for (int j = 0; j < target->arity; j++) {
            std::cout << raw_data_host[i*target->arity + j] << "    ";
        }
        std::cout << std::endl;
    }
    hipFreeHost(raw_data_host);
}

void print_memory_usage(){
    int num_gpus;
    size_t free, total;
    hipGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        hipSetDevice( gpu_id );
        int id;
        hipGetDevice( &id );
        hipMemGetInfo( &free, &total );
        std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
    }
}

tuple_size_t get_free_memory() {
    int num_gpus;
    size_t free, total;
    hipGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        hipSetDevice( gpu_id );
        int id;
        hipGetDevice( &id );
        hipMemGetInfo( &free, &total );
        return free;
    }
    return 0;
}

tuple_size_t get_total_memory() {
    int num_gpus;
    size_t free, total;
    hipGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        hipSetDevice( gpu_id );
        int id;
        hipGetDevice( &id );
        hipMemGetInfo( &free, &total );
        return total;
    }
    return 0;
}

void print_tuple_list(tuple_type* tuples, tuple_size_t rows, tuple_size_t arity) {
    tuple_type* tuples_host;
    hipMallocHost((void**) &tuples_host, rows * sizeof(tuple_type));
    hipMemcpy(tuples_host, tuples, rows * sizeof(tuple_type),
               hipMemcpyDeviceToHost);
    if (rows > 100) {
        rows = 100;
    }
    for (tuple_size_t i = 0; i < rows; i++) {
        tuple_type cur_tuple = tuples_host[i];
        
        tuple_type cur_tuple_host;
        hipMallocHost((void**) &cur_tuple_host, arity * sizeof(column_type));
        hipMemcpy(cur_tuple_host, cur_tuple, arity * sizeof(column_type),
                   hipMemcpyDeviceToHost);
        for (tuple_size_t j = 0; j < arity; j++) {
            std::cout << cur_tuple_host[j] << " ";
        }
        std::cout << std::endl;
    }
}
