cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(gdlog LANGUAGES CXX CUDA)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.04/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/gdlog_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

function(rapids_cpm_cuco)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.cuco")

  set(options)
  set(one_value INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have INSTALL_EXPORT_SET as this is need for rapids_cpm_find
  set(to_install OFF)
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(cuco version repository tag shallow exclude)

  set(to_exclude OFF)
  if(NOT to_install OR exclude)
    set(to_exclude ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(cuco ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(cuco ${version} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS cuco::cuco
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${to_exclude}
                  OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_EXAMPLES OFF"
                          "INSTALL_CUCO ${to_install}")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(cuco)

  # Propagate up variables that CPMFindPackage provide
  set(cuco_SOURCE_DIR "${cuco_SOURCE_DIR}" PARENT_SCOPE)
  set(cuco_BINARY_DIR "${cuco_BINARY_DIR}" PARENT_SCOPE)
  set(cuco_ADDED "${cuco_ADDED}" PARENT_SCOPE)
  set(cuco_VERSION ${version} PARENT_SCOPE)

endfunction()

rapids_cpm_cuco()

function(rapids_cpm_rmm)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.rmm")

  set(options)
  set(one_value INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(rmm version repository tag shallow exclude)
  set(to_exclude OFF)
  if(NOT _RAPIDS_INSTALL_EXPORT_SET OR exclude)
    set(to_exclude ON)
  endif()

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(rmm ${version} patch_command)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(rmm ${version} ${ARGN} ${_RAPIDS_UNPARSED_ARGUMENTS}
                  GLOBAL_TARGETS rmm::rmm
                  CPM_ARGS
                  GIT_REPOSITORY ${repository}
                  GIT_TAG ${tag}
                  GIT_SHALLOW ${shallow} ${patch_command}
                  EXCLUDE_FROM_ALL ${to_exclude}
                  OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(rmm)

  # Propagate up variables that CPMFindPackage provide
  set(rmm_SOURCE_DIR "${rmm_SOURCE_DIR}" PARENT_SCOPE)
  set(rmm_BINARY_DIR "${rmm_BINARY_DIR}" PARENT_SCOPE)
  set(rmm_ADDED "${rmm_ADDED}" PARENT_SCOPE)
  set(rmm_VERSION ${version} PARENT_SCOPE)

  # rmm creates the correct namespace aliases
endfunction()

rapids_cpm_rmm()

# CPMAddPackage(
#   NAME cuco
#   GITHUB_REPOSITORY NVIDIA/cuCollections
#   GIT_TAG dev
#   OPTIONS
#      "BUILD_TESTS OFF"
#      "BUILD_BENCHMARKS OFF"
#      "BUILD_EXAMPLES OFF"
# )

# find_package(rmm REQUIRED PATHS ${PROJECT_SOURCE_DIR}/thirdparty/rmm/cmake)
# find_package(rmm REQUIRED)

# CPMAddPackage(
#   NAME rmm
#   GITHUB_REPOSITORY rapidsai/rmm
#   GIT_TAG v24.04.00a
#   OPTIONS
#     "BUILD_TESTS OFF"
#     "BUILD_BENCHMARKS OFF"
#     "BUILD_EXAMPLES OFF"
# )

set(gpu_ra_src "${PROJECT_SOURCE_DIR}/src")
set(gpu_ra_include "${PROJECT_SOURCE_DIR}/include")
set(test_dir "${PROJECT_SOURCE_DIR}/test")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread --extended-lambda")
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
target_link_libraries(gpu_ra PUBLIC rmm::rmm cuco)
# target_link_libraries(gpu_ra PUBLIC rmm::rmm cuco)
target_compile_options(gpu_ra PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
set_target_properties(gpu_ra PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90" )
# set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES native)



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
set_target_properties(TC PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90")

# add_executable(PLEN ${source_path_lenght})
# target_link_libraries(PLEN gpu_ra)
# target_compile_features(PLEN PUBLIC cxx_std_20)
# set_target_properties(PLEN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CSPA ${source_cspa})
target_link_libraries(CSPA gpu_ra)
target_compile_features(CSPA PUBLIC cxx_std_20)
set_target_properties(CSPA PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90")

add_executable(SG ${source_file_sg})
target_link_libraries(SG gpu_ra)
target_compile_features(SG PUBLIC cxx_std_20)
set_target_properties(SG PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "80;89;90")

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
