#pragma once
#include "relation.cuh"
// test helper

void print_hashes(GHashRelContainer* target, const char *rel_name);

void print_tuple_rows(GHashRelContainer* target, const char *rel_name);

void print_tuple_raw_data(GHashRelContainer* target, const char *rel_name);

void print_memory_usage();

void print_tuple_list(tuple_type* tuples, tuple_size_t rows, tuple_size_t arity);
