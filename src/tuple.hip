
#include "../include/exception.h"
#include "../include/tuple.h"
#include <thrust/sort.h>

__global__ void extract_column(tuple_type *tuples, tuple_size_t rows,
                               tuple_size_t k, column_type *column) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        column[i] = tuples[i][k];
    }
}

__global__ void compute_hash(tuple_type *tuples, tuple_size_t rows,
                             tuple_size_t index_column_size,
                             column_type *hashes) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        hashes[i] = (column_type)prefix_hash(tuples[i], index_column_size);
    }
}

void sort_tuples(tuple_type *tuples, tuple_size_t rows, tuple_size_t arity,
                 tuple_size_t index_column_size, int grid_size,
                 int block_size) {

    column_type *col_tmp;
    hipMalloc((void **)&col_tmp, rows * sizeof(column_type));
    for (int k = arity - 1; k >= 0; k--) {
        extract_column<<<grid_size, block_size>>>(tuples, rows, k, col_tmp);
        checkHip(hipGetLastError());
        checkHip(hipDeviceSynchronize());
        thrust::stable_sort_by_key(thrust::device, col_tmp, col_tmp + rows,
                                   tuples);
        checkHip(hipDeviceSynchronize());
    }
    hipFree(col_tmp);
}

void sort_tuple_by_hash(tuple_type *tuples, tuple_size_t rows,
                        tuple_size_t arity, tuple_size_t index_column_size,
                        int grid_size, int block_size) {
    column_type *col_tmp;
    hipMalloc((void **)&col_tmp, rows * sizeof(column_type));
    compute_hash<<<grid_size, block_size>>>(tuples, rows, index_column_size,
                                            col_tmp);
    checkHip(hipGetLastError());
    checkHip(hipDeviceSynchronize());
    thrust::stable_sort_by_key(thrust::device, col_tmp, col_tmp + rows, tuples);
    checkHip(hipDeviceSynchronize());
    hipFree(col_tmp);
}
