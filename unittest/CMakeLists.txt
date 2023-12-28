
project(unittest LANGUAGES CXX CUDA)

add_executable(HISA_TEST src/hisa_test.cu)

target_include_directories(HISA_TEST PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(HISA_TEST PRIVATE gpu_ra)
target_compile_features(HISA_TEST PRIVATE cxx_std_20)

add_test(NAME HISA_TEST COMMAND HISA_TEST)

