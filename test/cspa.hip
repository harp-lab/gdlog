#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/exception.h"
#include "../include/lie.h"
#include "../include/print.h"
#include "../include/timer.h"

//////////////////////////////////////////////////////

long int get_row_size(const char *data_path) {
    std::ifstream f;
    f.open(data_path);
    char c;
    long i = 0;
    while (f.get(c))
        if (c == '\n')
            ++i;
    f.close();
    return i;
}

enum ColumnT { U64, U32 };

column_type *get_relation_from_file(const char *file_path, int total_rows,
                                    int total_columns, char separator,
                                    ColumnT ct) {
    column_type *data =
        (column_type *)malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                if (ct == U64) {
                    fscanf(data_file, "%lld%c", &data[(i * total_columns) + j],
                           &separator);
                } else {
                    fscanf(data_file, "%ld%c", &data[(i * total_columns) + j],
                           &separator);
                }
            } else {
                if (ct == U64) {
                    fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
                } else {
                    fscanf(data_file, "%ld", &data[(i * total_columns) + j]);
                }
            }
        }
    }
    return data;
}

//////////////////////////////////////////////////////////////////

__device__ void cp_2_1__1(tuple_type input, tuple_type outpt) {
    outpt[0] = input[0];
    outpt[1] = input[0];
};
__device__ tuple_copy_hook cp_2_1__1_device = cp_2_1__1;
__device__ void cp_2_1__2(tuple_type input, tuple_type outpt) {
    outpt[0] = input[1];
    outpt[1] = input[1];
};
__device__ tuple_copy_hook cp_2_1__2_device = cp_2_1__2;

__device__ void cp_2_1__1_2(tuple_type input, tuple_type outpt) {
    outpt[0] = input[1];
    outpt[1] = input[0];
};
__device__ tuple_copy_hook cp_2_1__1_2_device = cp_2_1__1_2;
__device__ void cp_2_1__2_1(tuple_type input, tuple_type outpt) {
    outpt[0] = input[0];
    outpt[1] = input[1];
};
__device__ tuple_copy_hook cp_2_1__2_1_device = cp_2_1__2_1;

__device__ void join_10_11(tuple_type inner, tuple_type outer,
                           tuple_type output) {
    output[1] = inner[1];
    output[0] = outer[1];
}
__device__ tuple_generator_hook join_10_11_device = join_10_11;

__device__ void join_01_11(tuple_type inner, tuple_type outer,
                           tuple_type output) {
    output[0] = inner[1];
    output[1] = outer[1];
}
__device__ tuple_generator_hook join_01_11_device = join_01_11;

////////////////////////////////////////////////////////////////

void analysis_bench(const char *dataset_path, int block_size, int grid_size) {
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    
    double spent_time;

    // load the input relation
    std::stringstream assign_fact_ss;
    assign_fact_ss << dataset_path << "/assign.facts";
    std::stringstream dereference_fact_ss;
    dereference_fact_ss << dataset_path << "/dereference.facts";
    // std::cout << assign_fact_ss.str() << std::endl;
    tuple_size_t assign_counts = get_row_size(assign_fact_ss.str().c_str());
    std::cout << "Input assign rows: " << assign_counts << std::endl;
    column_type *raw_assign_data = get_relation_from_file(
        assign_fact_ss.str().c_str(), assign_counts, 2, '\t', U32);
    std::cout << "reversing assign ... " << std::endl;
    column_type *raw_reverse_assign_data =
        (column_type *)malloc(assign_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < assign_counts; i++) {
        raw_reverse_assign_data[i * 2 + 1] = raw_assign_data[i * 2];
        raw_reverse_assign_data[i * 2] = raw_assign_data[i * 2 + 1];
    }

    tuple_size_t dereference_counts =
        get_row_size(dereference_fact_ss.str().c_str());
    std::cout << "Input dereference rows: " << dereference_counts << std::endl;
    column_type *raw_dereference_data = get_relation_from_file(
        dereference_fact_ss.str().c_str(), dereference_counts, 2, '\t', U32);
    std::cout << "reversing dereference ... " << std::endl;
    column_type *raw_reverse_dereference_data =
        (column_type *)malloc(dereference_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < dereference_counts; i++) {
        raw_reverse_dereference_data[i * 2 + 1] = raw_dereference_data[i * 2];
        raw_reverse_dereference_data[i * 2] = raw_dereference_data[i * 2 + 1];
    }

    timer.start_timer();
    
    Relation *assign_2__2_1 = new Relation();
    load_relation(assign_2__2_1, "assign_2__2_1", 2, raw_reverse_assign_data,
                  assign_counts, 1, 0, grid_size, block_size);

    Relation *dereference_2__1_2 = new Relation();
    load_relation(dereference_2__1_2, "dereference_2__1_2", 2,
                  raw_dereference_data, dereference_counts, 1, 0, grid_size,
                  block_size);
    Relation *dereference_2__2_1 = new Relation();
    load_relation(dereference_2__2_1, "dereference_2__2_1", 2,
                  raw_reverse_dereference_data, dereference_counts, 1, 0,
                  grid_size, block_size);
    timer.stop_timer();
    std::cout << "Build hash table time: " << timer.get_spent_time()
              << std::endl;

    // scc init
    Relation *value_flow_2__1_2 = new Relation();
    load_relation(value_flow_2__1_2, "value_flow_2__1_2", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);
    Relation *value_flow_2__2_1 = new Relation();
    load_relation(value_flow_2__2_1, "value_flow_2__2_1", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);

    Relation *memory_alias_2__1_2 = new Relation();
    load_relation(memory_alias_2__1_2, "memory_alias_2__1_2", 2, nullptr, 0, 1,
                  0, grid_size, block_size);
    Relation *memory_alias_2__2_1 = new Relation();
    load_relation(memory_alias_2__2_1, "memory_alias_2__2_1", 2, nullptr, 0, 1,
                  0, grid_size, block_size);

    timer.start_timer();
    time_point_begin = std::chrono::high_resolution_clock::now();
    LIE init_scc(grid_size, block_size);
    init_scc.add_relations(value_flow_2__1_2, false);
    init_scc.add_relations(value_flow_2__2_1, false);
    init_scc.add_relations(memory_alias_2__1_2, false);
    init_scc.add_relations(memory_alias_2__2_1, false);
    init_scc.add_relations(assign_2__2_1, true);
    tuple_copy_hook cp_2_1__1_host;
    checkHip(hipMemcpyFromSymbol(&cp_2_1__1_host, cp_2_1__1_device,
                         sizeof(tuple_copy_hook)));
    tuple_copy_hook cp_2_1__2_host;
    checkHip(hipMemcpyFromSymbol(&cp_2_1__2_host, cp_2_1__2_device,
                         sizeof(tuple_copy_hook)));
    tuple_copy_hook cp_2_1__1_2_host;
    checkHip(hipMemcpyFromSymbol(&cp_2_1__1_2_host, cp_2_1__1_2_device,
                         sizeof(tuple_copy_hook)));
    tuple_copy_hook cp_2_1__2_1_host;
    checkHip(hipMemcpyFromSymbol(&cp_2_1__1_host, cp_2_1__1_device,
                         sizeof(tuple_copy_hook)));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   cp_2_1__1_host, nullptr, grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   cp_2_1__2_host, nullptr, grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   cp_2_1__1_2_host, nullptr, grid_size,
                                   block_size));

    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, memory_alias_2__1_2,
                                   cp_2_1__1_host, nullptr, grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, memory_alias_2__1_2,
                                   cp_2_1__2_host, nullptr, grid_size,
                                   block_size));

    init_scc.add_ra(RelationalCopy(value_flow_2__1_2, DELTA, value_flow_2__2_1,
                                   cp_2_1__1_2_host, nullptr, grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(memory_alias_2__1_2, DELTA, memory_alias_2__2_1,
                                   cp_2_1__1_2_host, nullptr, grid_size,
                                   block_size));
    init_scc.fixpoint_loop();

    timer.stop_timer();
    time_point_end = std::chrono::high_resolution_clock::now();
    std::cout << "init scc time: " << timer.get_spent_time() << std::endl;
    std::cout << "init scc time (chono): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     time_point_end - time_point_begin)
                     .count()
              << std::endl;

    // scc analysis
    Relation *value_flow_forward_2__1_2 = new Relation();
    load_relation(value_flow_forward_2__1_2, "value_flow_forward_2__1_2", 2,
                  nullptr, 0, 1, 0, grid_size, block_size);

    Relation *value_flow_forward_2__2_1 = new Relation();
    load_relation(value_flow_forward_2__2_1, "value_flow_forward_2__2_1", 2,
                  nullptr, 0, 1, 0, grid_size, block_size);

    Relation *value_alias_2__1_2 = new Relation();
    value_alias_2__1_2->index_flag = false;
    load_relation(value_alias_2__1_2, "value_alias_2__1_2", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);

    Relation *tmp_rel_def = new Relation();
    tmp_rel_def->index_flag = false;
    load_relation(tmp_rel_def, "tmp_rel_def", 2, nullptr, 0, 1, 0, grid_size,
                  block_size);
    Relation *tmp_rel_ma1 = new Relation();
    tmp_rel_ma1->index_flag = false;
    load_relation(tmp_rel_ma1, "tmp_rel_ma1", 2, nullptr, 0, 1, 0, grid_size,
                  block_size, true);
     Relation *tmp_rel_ma2 = new Relation();
    tmp_rel_ma2->index_flag = false;
    load_relation(tmp_rel_ma2, "tmp_rel_ma2", 2, nullptr, 0, 1, 0, grid_size,
                  block_size, true);

    LIE analysis_scc(grid_size, block_size);

    analysis_scc.add_relations(assign_2__2_1, true);
    analysis_scc.add_relations(dereference_2__1_2, true);
    analysis_scc.add_relations(dereference_2__2_1, true);

    analysis_scc.add_relations(value_flow_2__1_2, false);
    analysis_scc.add_relations(value_flow_2__2_1, false);
    analysis_scc.add_relations(memory_alias_2__1_2, false);
    analysis_scc.add_relations(memory_alias_2__2_1, false);
    analysis_scc.add_relations(value_alias_2__1_2, false);

    // join order matters for temp!
    analysis_scc.add_tmp_relation(tmp_rel_def);
    analysis_scc.add_tmp_relation(tmp_rel_ma1);
    analysis_scc.add_tmp_relation(tmp_rel_ma2);

    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // join_vf_vfvf: ValueFlow(x, y) :- ValueFlow(x, z), ValueFlow(z, y).
    tuple_generator_hook join_10_11_host;
    checkHip(hipMemcpyFromSymbol(&join_10_11_host, join_10_11_device,
                         sizeof(tuple_generator_hook)));
    tuple_generator_hook join_01_11_host;
    checkHip(hipMemcpyFromSymbol(&join_01_11_host, join_01_11_device,
                         sizeof(tuple_generator_hook)));
    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, value_flow_2__2_1, DELTA,
                       value_flow_2__1_2, join_10_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));
    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__2_1, FULL, value_flow_2__1_2, DELTA,
                       value_flow_2__1_2, join_01_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    // join_va_vf_vf: ValueAlias(x, y) :- ValueFlow(z, x), ValueFlow(z, y).
    // v1
    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, value_flow_2__1_2, DELTA,
                       value_alias_2__1_2, join_01_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));
    // v2
    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, value_flow_2__1_2, DELTA,
                       value_alias_2__1_2, join_10_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    // join_vf_am: ValueFlow(x, y) :- Assign(x, z), MemoryAlias(z, y).
    analysis_scc.add_ra(
        RelationalJoin(assign_2__2_1, FULL, memory_alias_2__1_2, DELTA,
                       value_flow_2__1_2, join_01_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    // tmp_rel_def(z, x) :- Dereference(y, x), ValueAlias(y, z)
    analysis_scc.add_ra(
        RelationalJoin(dereference_2__1_2, FULL, value_alias_2__1_2, DELTA,
                       tmp_rel_def, join_10_11_host, nullptr, LEFT, grid_size,
                       block_size, join_detail));

    // WARNING: tmp relation can only in outer because it doesn't include
    // index!
    // join_ma_d_tmp: MemoryAlias(x, w) :- Dereference(z, w) , tmp_rel_def(z,x)
    analysis_scc.add_ra(
        RelationalJoin(dereference_2__1_2, FULL, tmp_rel_def, NEWT,
                       memory_alias_2__1_2, join_10_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    // ValueAlias(x,y) :- 
    //    ValueFlow(z,x),
    //    MemoryAlias(z,w),
    //    ValueFlow(w,y).
    // ValueFlow DELTA 1, 2 <> MemoryAlias FULL 1, 2 <> ValueFlow FULL 2, 1
    // ValueFlow FULL 1, 2 <> MemoryAlias DELTA 1, 2 <> ValueFlow FULL 2, 1
    // ValueFlow FULL 1, 2 <> MemoryAlias FULL 1, 2 <> ValueFlow DELTA 2, 1 
    // join_tmp_vf_ma : tmp_rel_ma(w, x) :- ValueFlow(z, x), MemoryAlias(z, w).
    // join_va_tmp_vf : ValueAlias(x, y) :- tmp_rel_ma(w, x), ValueFlow(w,y).
    // v1
    analysis_scc.add_ra(
        RelationalJoin(memory_alias_2__1_2, FULL , value_flow_2__1_2, DELTA,
                       tmp_rel_ma1, join_01_11_host, nullptr, LEFT, grid_size,
                       block_size, join_detail));
    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, memory_alias_2__1_2, DELTA,
                       tmp_rel_ma1, join_10_11_host, nullptr, LEFT, grid_size,
                       block_size, join_detail));

    analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, tmp_rel_ma1, NEWT,
                       value_alias_2__1_2, join_10_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    analysis_scc.add_ra(
        RelationalJoin(memory_alias_2__2_1, FULL , value_flow_2__1_2, DELTA,
                       tmp_rel_ma2, join_01_11_host, nullptr, LEFT, grid_size,
                       block_size, join_detail));
     analysis_scc.add_ra(
        RelationalJoin(value_flow_2__1_2, FULL, tmp_rel_ma2, NEWT,
                       value_alias_2__1_2, join_01_11_host, nullptr, LEFT,
                       grid_size, block_size, join_detail));

    analysis_scc.add_ra(RelationalACopy(value_flow_2__1_2, value_flow_2__2_1,
                                        cp_2_1__1_2_host, nullptr, grid_size,
                                        block_size));
    analysis_scc.add_ra(RelationalACopy(memory_alias_2__1_2, memory_alias_2__2_1,
                                        cp_2_1__1_2_host, nullptr, grid_size,
                                        block_size));
    time_point_begin = std::chrono::high_resolution_clock::now();
    timer.start_timer();
    analysis_scc.fixpoint_loop();
    // print_tuple_rows(value_flow_2__1_2->full, "value_flow_2__1_2");
    timer.stop_timer();
    time_point_end = std::chrono::high_resolution_clock::now();
    std::cout << "analysis scc time: " << timer.get_spent_time() << std::endl;
    std::cout << "analysis scc time (chono): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     time_point_end - time_point_begin)
                     .count()
              << std::endl;
    std::cout << "join detail: " << std::endl;
    std::cout << "compute size time:  " <<  join_detail[0] <<  std::endl;
    std::cout << "reduce + scan time: " <<  join_detail[1] <<  std::endl;
    std::cout << "fetch result time:  " <<  join_detail[2] <<  std::endl;
    std::cout << "sort time:          " <<  join_detail[3] <<  std::endl;
    std::cout << "build index time:   " <<  join_detail[5] <<  std::endl;
    std::cout << "merge time:         " <<  join_detail[6] <<  std::endl;
    std::cout << "unique time:        " << join_detail[4] + join_detail[7] <<  std::endl;
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    hipGetDevice(&device_id);
    hipDeviceGetAttribute(&number_of_sm, hipDeviceAttributeMultiprocessorCount,
                           device_id);
    int max_threads_per_block;
    hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, 0);
    std::cout << "num of sm " << number_of_sm << " num of thread per block " << max_threads_per_block << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");
    analysis_bench(argv[1], block_size, grid_size);
    return 0;
}
