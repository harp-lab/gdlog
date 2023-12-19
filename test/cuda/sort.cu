
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct transform_functor {
    int *raw;
    __host__ __device__ int *operator()(int i) { return raw + 2 * i; }
};

int main() {
    int vec_size = 2 << 25;
    int tuple_size = vec_size / 2;

    thrust::device_vector<int> d_vec(vec_size);

    // int data2[] = {4, 3, 2, 2, 5, 2, 1, 6, 1, 0};

    // int data1[] = {4, 2, 3, 1, 2, 6, 2, 1, 5, 0};
    // thrust::host_vector<int> d_vec_host1(data1, data1 + vec_size);
    thrust::host_vector<int> d_vec_host(vec_size);
    // thrust::host_vector<int> d_vec_host2(data2, data2 + vec_size);
    srand(13);
    thrust::generate(d_vec_host.begin(), d_vec_host.end(), rand);
    // thrust::host_vector<int> res1(vec_size);
    // thrust::host_vector<int> res2(vec_size);

    d_vec = d_vec_host;
    std::cout << "d_vec size : " << d_vec.size() << std::endl;
    // generate ptrs vector every 2 elements
    thrust::device_vector<int *> d_ptrs_vec(d_vec.size() / 2);
    auto start = std::chrono::high_resolution_clock::now();
    thrust::counting_iterator<int> iter_begin(0);
    thrust::counting_iterator<int> iter_end(d_ptrs_vec.size());
    thrust::transform(thrust::device, iter_begin, iter_end, d_ptrs_vec.begin(),
                      transform_functor{d_vec.data().get()});
    std::cout << "d_ptrs_vec size : " << d_ptrs_vec.size() << std::endl;
    // sort ptrs vector
    // get time of sorting using chrono
    thrust::sort(d_ptrs_vec.begin(), d_ptrs_vec.end(),
                 [] __device__(int *a, int *b) -> bool {
                     if (a[0] < b[0]) {
                         return true;
                     } else if (a[0] > b[0]) {
                         return false;
                     } else {
                         return a[1] < b[1];
                     }
                 });
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Sorting time 1 : " << diff.count() << " s\n";

    // for (int i = 0; i < d_ptrs_vec.size(); i++) {
    //     int host_tp[2];
    //     cudaMemcpy(host_tp, d_ptrs_vec[i], 2 * sizeof(int),
    //                cudaMemcpyDeviceToHost);
    //     std::cout << host_tp[0] << " " << host_tp[1] << std::endl;
    // }

    d_vec = d_vec_host;
    // transpose 2d array d_vec
    thrust::device_vector<int> d_vec_transposed(d_vec.size());
    // thrust::device_vector<int> d_vec_transposed = d_vec_host2;
    thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(d_vec_transposed.size()),
                     [raw = d_vec.data().get(),
                      raw_transposed = d_vec_transposed.data().get(),
                      tp_size = tuple_size] __device__(int i) {
                         int row = i / 2;
                         int col = i % 2;
                         raw_transposed[col * 2 + row] = raw[row * 2 + col];
                     });
    thrust::device_vector<int> idxs(tuple_size);
    thrust::device_vector<int> tmp(tuple_size);
    auto start2 = std::chrono::high_resolution_clock::now();
    thrust::counting_iterator<int> iter_begin2(0);
    thrust::counting_iterator<int> iter_end2(idxs.size());

    thrust::transform(thrust::device, iter_begin2, iter_end2, idxs.begin(),
                      thrust::identity<int>());
    thrust::transform(thrust::device, idxs.begin(), idxs.end(), tmp.begin(),
                      [raw = d_vec_transposed.begin() +
                             tuple_size] __device__(int i) { return raw[i]; });
    thrust::sort_by_key(thrust::device, tmp.begin(), tmp.end(), idxs.begin());
    thrust::transform(
        thrust::device, idxs.begin(), idxs.end(), tmp.begin(),
        [raw = d_vec_transposed.begin()] __device__(int i) { return raw[i]; });
    // d_vec_transposed = d_vec_host2;
    thrust::sort_by_key(thrust::device, tmp.begin(), tmp.end(), idxs.begin());
    // d_vec_transposed = d_vec_host2;

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = end2 - start2;
    std::cout << "Sorting time 2 : " << diff2.count() << " s\n";

    // res2 = idxs;
    // for (auto &i : res2) {
    //     std::cout << data2[i] << " " << data2[i + tuple_size] << std::endl;
    // }
    // d_vec_host2 = d_vec_transposed;
    // for (auto &i : res2) {
    //     std::cout << i << " ";
    // }

    int data1[] = {1, 2, 4,
                   2, 4, 7};
    int data2[] = {1, 1, 3,
                   5, 7, 3};
    vec_size = 6;
    int tp_size1 = 3;
    int tp_size2 = 3;
    thrust::host_vector<int> d_vec_merge1(data1, data1 + vec_size);
    thrust::host_vector<int> d_vec_merge2(data2, data2 + vec_size);
    thrust::device_vector<int> d_vec_merge1_dev = d_vec_merge1;
    thrust::device_vector<int> d_vec_merge2_dev = d_vec_merge2;
    thrust::device_vector<int> d_vec_merge_res(tp_size1 + tp_size2);
    thrust::device_vector<int> d_vec_merge_idx1(tp_size1);
    thrust::device_vector<int> d_vec_merge_idx2(tp_size2);
    thrust::device_vector<int> d_vec_merge_tmp1(tp_size1);
    thrust::device_vector<int> d_vec_merge_tmp2(tp_size2);
    thrust::device_vector<bool> d_vec_merge_nonsence(tp_size1 + tp_size2);

    thrust::counting_iterator<int> iter_begin_merge1(0);
    thrust::counting_iterator<int> iter_end_merge1(tp_size1);
    thrust::transform(thrust::device, iter_begin_merge1, iter_end_merge1,
                      d_vec_merge_idx1.begin(), thrust::identity<int>());
    thrust::counting_iterator<int> iter_begin_merge2(tp_size1);
    thrust::counting_iterator<int> iter_end_merge2(tp_size1 + tp_size2);
    thrust::transform(thrust::device, iter_begin_merge2, iter_end_merge2,
                      d_vec_merge_idx2.begin(),
                      thrust::identity<int>());

    thrust::transform(thrust::device, d_vec_merge_idx1.begin(),
                      d_vec_merge_idx1.end(), d_vec_merge_tmp1.begin(),
                      [raw = d_vec_merge1_dev.begin() +
                             tp_size1] __device__(int i) { return raw[i]; });
    thrust::transform(thrust::device, d_vec_merge_idx2.begin(),
                      d_vec_merge_idx2.end(), d_vec_merge_tmp2.begin(),
                      [raw = d_vec_merge2_dev.begin() +
                             tp_size2, tp_size1] __device__(int i) { return raw[i-tp_size1]; });
    thrust::merge_by_key(thrust::device,
                         d_vec_merge_tmp1.begin(), d_vec_merge_tmp1.end(),
                         d_vec_merge_tmp2.begin(), d_vec_merge_tmp2.end(),
                         d_vec_merge_idx1.begin(), d_vec_merge_idx2.begin(),
                         d_vec_merge_nonsence.begin(), d_vec_merge_res.begin());

    thrust::transform(
        thrust::device, d_vec_merge_idx1.begin(), d_vec_merge_idx1.end(),
        d_vec_merge_tmp1.begin(),
        [raw = d_vec_merge1_dev.begin()] __device__(int i) { return raw[i]; });
    thrust::transform(
        thrust::device, d_vec_merge_idx2.begin(), d_vec_merge_idx2.end(),
        d_vec_merge_tmp2.begin(),
        [raw = d_vec_merge2_dev.begin(), tp_size1] __device__(int i) { return raw[i-tp_size1]; });
    thrust::merge_by_key(thrust::device,
                         d_vec_merge_tmp1.begin(), d_vec_merge_tmp1.end(),
                         d_vec_merge_tmp2.begin(), d_vec_merge_tmp2.end(),
                         d_vec_merge_idx1.begin(), d_vec_merge_idx2.begin(),
                         d_vec_merge_nonsence.begin(), d_vec_merge_res.begin());
    
    // print res
    thrust::host_vector<int> res(vec_size);
    res = d_vec_merge_res;
    for (auto &i : res) {
        if (i < 3) {
            std::cout << d_vec_merge1[i] << " " <<  d_vec_merge1[i+tp_size1] << std::endl;
        } else {
            std::cout << d_vec_merge2[i - 3] << " "
                      << d_vec_merge2[i] << std::endl;
        }
    }
    return 0;
}
