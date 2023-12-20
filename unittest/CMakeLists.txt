
project(unittest LANGUAGES CXX CUDA)

add_executable(HISA_TEST src/hisa_test.cu)

target_link_libraries(HISA_TEST PRIVATE gpu_ra)

add_test(NAME HISA_TEST COMMAND HISA_TEST)

