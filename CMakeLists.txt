cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(tc_gpu LANGUAGES CXX CUDA)

find_package(rmm REQUIRED)

include(${PROJECT_SOURCE_DIR}/CPM/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
set(test_dir "${PROJECT_SOURCE_DIR}/test")

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread --extended-lambda")
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # does not produce the json file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works


file(GLOB source_file_gpu_ra
    "${gpu_ra_include}/exception.cuh"
    "${gpu_ra_include}/dynamic_dispatch.h"
    "${gpu_ra_include}/print.cuh"
    "${gpu_ra_include}/relation.cuh"
    "${gpu_ra_include}/relational_algebra.cuh"
    "${gpu_ra_include}/timer.cuh"
    "${gpu_ra_include}/tuple.cuh"
    "${gpu_ra_include}/hisa.hpp"
    "${gpu_ra_include}/lie.cuh"
    "${gpu_ra_include}/hash.h"
    "${gpu_ra_src}/tuple.cu"
    "${gpu_ra_src}/print.cu"
    "${gpu_ra_src}/relation.cu"
    "${gpu_ra_src}/lie.cu"
    "${gpu_ra_src}/copy.cu"
    "${gpu_ra_src}/acopy.cu"
    "${gpu_ra_src}/join.cu"
)

file(GLOB source_file_tc
    "${test_dir}/tc.cu"
)

file(GLOB source_path_lenght
    "${test_dir}/path_length.cu"
)

file(GLOB source_cspa
    "${test_dir}/cspa.cu"
)

file(GLOB source_file_sg
    "${test_dir}/sg.cu"
)

# file(GLOB source_file_test
# "${test_dir}/sort.cu"
# )
file(GLOB source_file_merge
    "${test_dir}/merge.cu"
)

file(GLOB source_file_dedup
    "${test_dir}/dedup.cu"
)

add_library(gpu_ra "${source_file_gpu_ra}")
target_compile_features(gpu_ra PUBLIC cxx_std_20)
target_link_libraries(gpu_ra PUBLIC rmm::rmm cuco CUDA::cudart)
target_compile_options(gpu_ra PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




# using test
include(CTest)
# include(Catch)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(unittest)
endif ()
enable_testing()



add_executable(TC ${source_file_tc})
target_link_libraries(TC gpu_ra)
target_compile_features(TC PUBLIC cxx_std_20)
set_target_properties(TC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(PLEN ${source_path_lenght})
# target_link_libraries(PLEN gpu_ra)
# target_compile_features(PLEN PUBLIC cxx_std_20)
# set_target_properties(PLEN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CSPA ${source_cspa})
target_link_libraries(CSPA gpu_ra)
target_compile_features(CSPA PUBLIC cxx_std_20)
set_target_properties(CSPA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(SG ${source_file_sg})
target_link_libraries(SG gpu_ra)
target_compile_features(SG PUBLIC cxx_std_20)
set_target_properties(SG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(TEST ${source_file_test})
# target_link_libraries(TEST gpu_ra)
# target_compile_features(TEST PUBLIC cxx_std_20)
# set_target_properties(TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(MERGE ${source_file_merge})
# target_link_libraries(MERGE gpu_ra)
# target_compile_features(MERGE PUBLIC cxx_std_20)
# set_target_properties(MERGE PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(DEDUP ${source_file_dedup})
# target_link_libraries(DEDUP gpu_ra)
# target_compile_features(DEDUP PUBLIC cxx_std_20)
# set_target_properties(DEDUP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
