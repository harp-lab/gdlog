/**
 * test data structure
 */

#include "rmm/device_vector.hpp"
#include <cstdint>
#include <iostream>
#include <rmm/device_vector.hpp>
#include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include "../../include/hisa.hpp"
#include "../../include/print.cuh"
#include "../../include/timer.cuh"

const int TEST_RELATION_DATA1[] = {1, 3, 3, 8, 8, 3, 7, 2, 3,
                                   1, 5, 2, 2, 3, 1, 1, 3, 2};
const int data1_size = 9;

const int TEST_RELATION_DATA2[] = {1, 2, 3, 3, 3, 5, 8, 9,
                                   8, 3, 3, 4, 6, 2, 2, 1};
const int data2_size = 8;

// test build of HISA
bool test_build() {
    // build a HISA
    std::cout << "test build" << std::endl;
    KernelTimer timer;
    HISA<int, rmm::device_vector<int>, uint32_t> hisa_test(2, 1);
    // thrust::host_vector<int> h_relation1(TEST_RELATION_DATA1,
    // TEST_RELATION_DATA1 + 16);
    rmm::device_vector<int> d_relation1(TEST_RELATION_DATA1,
                                        TEST_RELATION_DATA1 + data1_size * 2);

    timer.start_timer();
    hisa_test.build(d_relation1, data1_size);
    timer.stop_timer();
    std::cout << "build time: " << timer.get_spent_time() << std::endl;

    print_hisa(hisa_test, "test");
    return true;
}

int main() {
    test_build();
    return 1;
}
