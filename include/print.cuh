#pragma once
#include "hisa.hpp"
#include "relation.cuh"
#include "tuple.cuh"

#include <thrust/host_vector.h>
// test helper

void print_hashes(GHashRelContainer* target, const char *rel_name);

void print_tuple_rows(GHashRelContainer* target, const char *rel_name, bool sort_flag = true);

void print_tuple_raw_data(GHashRelContainer* target, const char *rel_name);

void print_memory_usage();

// void print_tuple_list(tuple_type* tuples, tuple_size_t rows, tuple_size_t arity);

tuple_size_t get_free_memory();

tuple_size_t get_total_memory();


template <typename col_type, typename container_type,
          typename column_index_type>
void print_hisa(HISA<col_type, container_type, column_index_type> &hisa,
                char *msg) {
    std::cout << "HISA >>> " << msg << std::endl;
    std::cout << "Total tuples counts:  " << hisa.total_row_size << std::endl;
    // copy to host
    thrust::host_vector<col_type> host_indices(hisa.total_row_size);
    thrust::copy(hisa.index_container->lex_offset.begin(),
                 hisa.index_container->lex_offset.end(), host_indices.begin());
    thrust::host_vector<col_type>* host_cols =
        new thrust::host_vector<col_type>[hisa.arity];

    for (int i = 0; i < hisa.arity; i++) {
        host_cols[i].resize(hisa.total_row_size);
        // host_cols[i] = hisa.data_containers[i].data;
        thrust::copy(hisa.data_containers[i].data.begin(),
                     hisa.data_containers[i].data.end(), host_cols[i].begin());
    }

    for (auto &idx : host_indices) {
        std::cout << idx << ":\t";
        for (int i = 0; i < hisa.arity; i++) {
            std::cout << host_cols[i][idx] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "end <<<" << std::endl;

    delete[] host_cols;
}
