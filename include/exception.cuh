#pragma once
#include <assert.h>
// #include <cuda_runtime.h>
#include <iostream>

#define checkCuda(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}
