cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(tc_gpu LANGUAGES CXX HIP)

set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
set(test_dir "${PROJECT_SOURCE_DIR}/test")

file(GLOB source_file_gpu_ra 
    "${gpu_ra_include}/exception.h"
    "${gpu_ra_include}/dynamic_dispatch.h"
    "${gpu_ra_include}/print.h"
    "${gpu_ra_include}/relation.h"
    "${gpu_ra_include}/relational_algebra.h"
    "${gpu_ra_include}/timer.h"
    "${gpu_ra_include}/tuple.h"
    "${gpu_ra_include}/lie.h"
    "${gpu_ra_src}/tuple.hip"
    "${gpu_ra_src}/print.hip"
    "${gpu_ra_src}/relation.hip"
    "${gpu_ra_src}/lie.hip"
    "${gpu_ra_src}/copy.hip"
    "${gpu_ra_src}/acopy.hip"
    "${gpu_ra_src}/join.hip"
)

file(GLOB source_file_tc
    "${test_dir}/tc.hip"
)

file(GLOB source_path_lenght
    "${test_dir}/path_length.hip"
)

file(GLOB source_cspa
    "${test_dir}/cspa.hip"
)

file(GLOB source_file_datastructure
    "${test_dir}/datastructure.hip"
)

file(GLOB source_file_sg
    "${test_dir}/sg.hip"
)

file(GLOB source_file_test
    "${test_dir}/sort.hip"
)

file(GLOB source_file_merge
    "${test_dir}/merge.hip"
)

add_library(gpu_ra "${source_file_gpu_ra}")
target_compile_features(gpu_ra PUBLIC cxx_std_20)
set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(TC ${source_file_tc})
target_link_libraries(TC gpu_ra)
target_compile_features(TC PUBLIC cxx_std_20)
set_target_properties(TC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(PLEN ${source_path_lenght})
target_link_libraries(PLEN gpu_ra)
target_compile_features(PLEN PUBLIC cxx_std_20)
set_target_properties(PLEN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CSPA ${source_cspa})
target_link_libraries(CSPA gpu_ra)
target_compile_features(CSPA PUBLIC cxx_std_20)
set_target_properties(CSPA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(DATASTRUCTURE ${source_file_datastructure})
target_link_libraries(DATASTRUCTURE gpu_ra)
target_compile_features(DATASTRUCTURE PUBLIC cxx_std_20)
set_target_properties(DATASTRUCTURE PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(SG ${source_file_sg})
target_link_libraries(SG gpu_ra)
target_compile_features(SG PUBLIC cxx_std_20)
set_target_properties(SG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(TEST ${source_file_test})
target_link_libraries(TEST gpu_ra)
target_compile_features(TEST PUBLIC cxx_std_20)
set_target_properties(TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(MERGE ${source_file_merge})
target_link_libraries(MERGE gpu_ra)
target_compile_features(MERGE PUBLIC cxx_std_20)
set_target_properties(MERGE PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
