
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct transform_functor {
    int *raw;
    __host__ __device__ int *operator()(int i) { return raw + 2 * i; }
};

int main() {
    std::cout << "Hello World!\n";
    int vec_size = 1 << 28;
    int tuple_size = vec_size / 2;
    thrust::device_vector<int> d_vec(vec_size);
    thrust::host_vector<int> d_vec_host(vec_size);
    srand(13);
    thrust::generate(d_vec_host.begin(), d_vec_host.end(), rand);
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

    // transpose 2d array d_vec
    thrust::device_vector<int> d_vec_transposed(d_vec.size());
    thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(d_vec_transposed.size()),
                     [raw = d_vec.data().get(),
                      raw_transposed = d_vec_transposed.data().get(),
                      tp_size = tuple_size] __device__(int i) {
                         int row = i / 2;
                         int col = i % 2;
                         raw_transposed[col * 2 + row] = raw[row * 2 + col];
                     });
    thrust::device_vector<int> idxs(d_ptrs_vec.size());
    auto start2 = std::chrono::high_resolution_clock::now();
    thrust::counting_iterator<int> iter_begin2(0);
    thrust::counting_iterator<int> iter_end2(idxs.size());
    thrust::transform(thrust::device, iter_begin2, iter_end2, idxs.begin(),
                      thrust::identity<int>());
    thrust::stable_sort_by_key(
        thrust::device, d_vec_transposed.begin() + tuple_size,
        d_vec_transposed.end(), idxs.begin() + tuple_size);
    thrust::stable_sort_by_key(thrust::device, d_vec_transposed.begin(),
                               d_vec_transposed.begin() + tuple_size,
                               idxs.begin());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = end2 - start2;
    std::cout << "Sorting time 2 : " << diff2.count() << " s\n";

    return 0;
}
