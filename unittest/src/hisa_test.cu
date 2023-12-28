/**
 * test data structure
 */

#include "rmm/device_vector.hpp"
#include <cstdint>
#include <iostream>
#include <rmm/device_vector.hpp>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include "../../include/hisa.hpp"
#include "../../include/print.cuh"
#include "../../include/timer.cuh"

const int TEST_RELATION_DATA1[] = {1, 3, 3, 8, 8, 3, 7, 2, 3,
                                   1, 5, 2, 2, 3, 1, 1, 3, 2};
const int data1_size = 9;
const int ORDERED_DATA1[] = {1, 2, 3, 3, 3, 7, 8, 8,
                             1, 3, 1, 2, 5, 1, 2, 3};

const int TEST_RELATION_DATA2[] = {1, 2, 3, 3, 3, 5, 8, 9,
                                   8, 3, 3, 4, 6, 2, 2, 1};
const int data2_size = 8;

// test build of HISA
bool test_build() {
    // build a HISA
    std::cout << "test build" << std::endl;
    KernelTimer timer;
    HISA<2> hisa_test(1);
    // thrust::host_vector<int> h_relation1(TEST_RELATION_DATA1,
    // TEST_RELATION_DATA1 + 16);
    rmm::device_vector<int> d_relation1(TEST_RELATION_DATA1,
                                        TEST_RELATION_DATA1 + data1_size * 2);

    timer.start_timer();
    hisa_test.build(d_relation1, data1_size);
    timer.stop_timer();
    std::cout << "build time: " << timer.get_spent_time() << std::endl;

    // print_hisa(hisa_test, "test");
    // deduplication count check
    if (hisa_test.total_row_size != 8) {
        std::cout << "total row count error, deduplication failed" << std::endl;
        return false;
    }
    // HISAHost<> hisa_host(hisa_test);

    // check if sorted result is correct
    // for (int i = 0; i < hisa_host.total_row_count; i++) {
    //     if
    //     (hisa_host.data_container[0].data[hisa_host->index_container.lex_offset[i]]
    //     !=
    //         ORDERED_DATA1[i]) {
    //         std::cout << "sorted result error exprected " << ORDERED_DATA1[i]
    //         << " but get "
    //                   <<
    //                   hisa_host.data_container[0].data[hisa_host->index_container.lex_offset[i]]
    //                   << std::endl;
    //         return false;
    //     }
    //     if
    //     (hisa_host.data_container[1].data[hisa_host->index_container.lex_offset[i]]
    //     !=
    //         ORDERED_DATA1[i+8]) {
    //         std::cout << "sorted result error exprected " <<
    //         ORDERED_DATA1[i+8] << " but get "
    //                   <<
    //                   hisa_host.data_container[1].data[hisa_host->index_container.lex_offset[i]]
    //                   << std::endl;
    //         return false;
    //     }
    // }

    return true;
}

bool test_merge() {
    // build a HISA
    std::cout << "test merge 1" << std::endl;
    KernelTimer timer;
    HISA<2> hisa_test1(1);
    HISA<2> hisa_test2(1);
    // thrust::host_vector<int> h_relation1(TEST_RELATION_DATA1,
    // TEST_RELATION_DATA1 + 16);
    rmm::device_vector<int> d_relation1(TEST_RELATION_DATA1,
                                        TEST_RELATION_DATA1 + data1_size * 2);
    rmm::device_vector<int> d_relation2(TEST_RELATION_DATA2,
                                        TEST_RELATION_DATA2 + data2_size * 2);

    timer.start_timer();
    hisa_test1.build(d_relation1, data1_size);
    hisa_test2.build(d_relation2, data2_size);
    print_hisa(hisa_test1, "test1 before merge");
    // print_hisa(hisa_test2, "test2 before merge");
    timer.stop_timer();
    std::cout << "build time: " << timer.get_spent_time() << std::endl;

    hisa_test1.difference_and_merge(hisa_test2);
    print_hisa(hisa_test1, "test1 after merge");
    print_hisa(hisa_test2, "test2 after merge");
}

bool test_merge2() {
    // build a HISA
    std::cout << "test merge 2" << std::endl;
    KernelTimer timer;
    HISA<2> hisa_test1(1);
    HISA<2> hisa_test2(1);
    HISA<2> hisa_diff(1);
    // thrust::host_vector<int> h_relation1(TEST_RELATION_DATA1,
    // TEST_RELATION_DATA1 + 16);
    rmm::device_vector<int> d_relation1(TEST_RELATION_DATA1,
                                        TEST_RELATION_DATA1 + data1_size * 2);
    rmm::device_vector<int> d_relation2(TEST_RELATION_DATA2,
                                        TEST_RELATION_DATA2 + data2_size * 2);

    timer.start_timer();
    hisa_test1.build(d_relation1, data1_size);
    hisa_test2.build(d_relation2, data2_size);
    print_hisa(hisa_test1, "test1 before merge");
    // print_hisa(hisa_test2, "test2 before merge");
    timer.stop_timer();
    std::cout << "build time: " << timer.get_spent_time() << std::endl;

    hisa_test1.difference(hisa_test2, hisa_diff);
    print_hisa(hisa_diff, "diff res after merge");
    hisa_test1.merge_unique(hisa_diff);
    print_hisa(hisa_test1, "test1 after merge");
}

int main() {
    bool build_res = test_build();
    if (!build_res) {
        std::cout << "build failed" << std::endl;
        return 1;
    }
    bool merge_res = test_merge();
    test_merge2();
    return 0;
}
