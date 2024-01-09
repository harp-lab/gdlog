// write a test for thrust::set_difference

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>

int main() {

    int data1[] = {1, 2, 3, 3, 3, 7, 8, 8,
                   1, 3, 5, 1, 2, 1, 2, 3};

    int data2[] = {1, 2, 3, 3, 3, 5, 8, 9,
                   8, 3, 3, 4, 6, 2, 2, 1};

    int data1_size = sizeof(data1) / sizeof(int);
    int data2_size = sizeof(data2) / sizeof(int);
    int data1_tp_size = data1_size / 2;
    int data2_tp_size = data2_size / 2;
    thrust::host_vector<int> h_data1(data1, data1 + data1_tp_size);
    thrust::host_vector<int> h_data2(data2, data2 + data2_tp_size);

    thrust::host_vector<int> data1_idx(data1_tp_size);
    thrust::sequence(data1_idx.begin(), data1_idx.end());

    thrust::host_vector<int> data2_idx(data2_tp_size);
    thrust::sequence(data2_idx.begin(), data2_idx.end());

    thrust::host_vector<int> h_result_k(data1_tp_size + data2_tp_size);
    thrust::host_vector<int> h_result_v(data1_tp_size + data2_tp_size);

    thrust::host_vector<int> tmp1;
    thrust::host_vector<int> tmp2;

    
    auto res_end = thrust::set_intersection_by_key(
        h_data1.begin(), h_data1.begin() + data1_tp_size,
        h_data2.begin(), h_data2.begin() + data2_tp_size,
        data1_idx.begin(),
        h_result_k.begin(), h_result_v.begin());

    // remove none intersection elements
    tmp1.resize(res_end.second - h_result_v.begin());
    thrust::transform(
        h_result_v.begin(), h_result_v.end(),
        tmp1.begin(),
        [data1_ptr = h_data1.begin()
         tp_size = tp] __host__ __device__(int v) { 
            return data1_ptr[v];
        });
    tmp2.resize(res_end.first - h_result_k.begin());
    thrust::transform(
        h_result_v.begin(), h_result_v.end(),
        tmp2.begin(),
        [data2_ptr = h_data2.begin()] __host__ __device__(int v) { 
            return data2_ptr[2];
        });
    
    auto res_end = thrust::set_intersection_by_key(
        tmp1.begin(), tmp1.end(),
        h_data2.begin(), h_data2.begin() + data2_tp_size,
        data1_idx.begin(),
        h_result_k.begin(), h_result_v.begin());


    // thrust::set_intersection_by_key(
    //     h_data1.begin() + data1_tp_size, h_data1.end(),
    //     h_data2.begin() + data2_tp_size, h_data2.end(),
    //     data1_idx.begin(),
    //     h_result_k.begin(), h_result_v.begin());

    for (auto i = 0; i < h_result_k.size(); ++i) {
        std::cout << h_result_k[i] << " " << h_result_v[i] << std::endl;
    }

    return 0;
}
