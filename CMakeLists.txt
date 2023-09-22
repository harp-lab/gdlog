cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(tc_gpu LANGUAGES CXX CUDA)

set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
set(test_dir "${PROJECT_SOURCE_DIR}/test")

file(GLOB source_file_gpu_ra 
    "${gpu_ra_include}/exception.cuh"
    "${gpu_ra_include}/dynamic_dispatch.h"
    "${gpu_ra_include}/print.cuh"
    "${gpu_ra_include}/relation.cuh"
    "${gpu_ra_include}/relational_algebra.cuh"
    "${gpu_ra_include}/timer.cuh"
    "${gpu_ra_include}/tuple.cuh"
    "${gpu_ra_include}/lie.cuh"
    "${gpu_ra_src}/print.cu"
    "${gpu_ra_src}/relation.cu"
    "${gpu_ra_src}/lie.cu"
    "${gpu_ra_src}/copy.cu"
    "${gpu_ra_src}/join.cu"
)

file(GLOB source_file_tc
    "${test_dir}/tc.cu"
)

add_library(gpu_ra "${source_file_gpu_ra}")
target_compile_features(gpu_ra PUBLIC cxx_std_20)
set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(TC ${source_file_tc})
target_link_libraries(TC gpu_ra)
target_compile_features(TC PUBLIC cxx_std_20)
set_target_properties(TC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
