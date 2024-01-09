
project(unittest LANGUAGES CXX CUDA)

add_executable(HISA_TEST src/hisa_test.cu)

target_include_directories(HISA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(HISA_TEST PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    --expt-relaxed-constexpr
    --default-stream per-thread
    --expt-extended-lambda>)
target_link_libraries(HISA_TEST PRIVATE gpu_ra)
target_compile_features(HISA_TEST PRIVATE cxx_std_20)

add_test(NAME HISA_TEST COMMAND HISA_TEST)

