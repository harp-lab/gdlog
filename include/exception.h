#pragma once
#include <assert.h>
// #include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <iostream>

#define checkHip(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != hipSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
                line);
//        if (abort) {
//            hipDeviceReset();
//            exit(code);
//        }
    }
}
