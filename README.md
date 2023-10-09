## SLOG GPU Backend
Some prototypes for slog's gpu backend

## Datasets
- Datasets are listed in [data folder](data).

## Dependencies
### Hardware
- The complete benchmark of the CUDA-based transitive closure computation experiment can be executed on an Nvidia A100 GPU with a minimum of 40 GB GPU memory. The ThetaGPU single-GPU node is a suitable choice.
- Partial benchmarks can be run on other Nvidia GPUs, but they may result in program termination for certain datasets due to limited GPU memory, leading to an instance of the `std::bad_alloc: cudaErrorMemoryAllocation: out of memory` error.

### NVIDIA CUDA Toolkit (version 11.4.2 or later)
- Download and install the NVIDIA CUDA Toolkit from the NVIDIA website: [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Follow the installation instructions for your operating system. Make sure to install version 11.4.2 or later.
### CMake 
- Download and install CMake(version 3.9 or later) from the CMake website: [https://cmake.org/download/](https://cmake.org/download/)
## Thrust
- need apply patch https://github.com/NVIDIA/thrust/pull/1832/files to fix integer overflow in `thrust::reduce`

## Transitive Closure Computation
- Transitive closure computation is a fundamental operation in graph analytics and relational algebra.
- We present a CUDA-based implementation of transitive closure computation that is optimized for sparse graphs.
- Build and run instructions are provided below:
```shell
cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S./ -B./build 
cd build
make
```
This will build the `TC` executable using the nvcc compiler.
- The `TC` executable takes a single argument, which is the path to the input file containing the graph data. The input file should be in the following format:
```shell
./TC ../data/data_5.txt
```
### Run instructions for Polaris
- Run using Interactive node:
```shell
ssh <USERNAME>@polaris.alcf.anl.gov
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A dist_relational_alg
module load gcc
cd slog-gpu-backend
git fetch
git reset --hard origin/main
rm -rf build
module purge
module load gcc
module load cmake
module load cudatoolkit-standalone
cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S./ -B./build 
cd build
make
./TC ../data/data_5.txt
```
- Transfer a file from local machine to Polaris:
```shell
scp data_68993773.txt arsho@polaris.alcf.anl.gov:/home/arsho/slog-gpu-backend/data/
```
### (Optional) Memory check:
- After creating the build folder and `TC` executable, run the following commands to check for memory leaks and errors:
```shell
cuda-memcheck ./TC ../data/data_7035.txt
========= CUDA-MEMCHECK
...
TC time: 48.691
========= ERROR SUMMARY: 0 errors
compute-sanitizer ./TC ../data/data_7035.txt
========= COMPUTE-SANITIZER
...
TC time: 0.668892
========= ERROR SUMMARY: 0 errors
```

### Out of memory error:
- If the program terminates with the following error, it means that the GPU memory is not sufficient to store the graph data:
```shell
arsho::x3004c0s7b0n0 { ~/slog-gpu-backend/build }-> ./TC ../data/data_68993773.txt
num of sm 108
using 18446744073709551615 as empty hash entry
Input graph rows: 68993773
reversing graph ... 
finish reverse graph.
edge size 68993773
Build hash table time: 0.551132
start lie .... 
GPUassert: an illegal memory access was encountered /home/arsho/slog-gpu-backend/src/relational_algebra.cu 78
```

### Examples
a TC example
```
    Relation *edge_2__2_1 = new Relation();
    Relation *path_2__1_2 = new Relation();

    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);

    LIE tc_scc(grid_size, block_size);
    tc_scc.add_relations(edge_2__2_1, true);
    tc_scc.add_relations(path_2__1_2, false);
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tuple_generator_hook reorder_path_host;
    cudaMemcpyFromSymbol(&reorder_path_host, reorder_path_device,
                         sizeof(tuple_generator_hook));
    tuple_copy_hook cp_1_host;
    cudaMemcpyFromSymbol(&cp_1_host, cp_1_device, sizeof(tuple_copy_hook));
    tc_scc.add_ra(RelationalJoin(edge_2__2_1, FULL, path_2__1_2, DELTA,
                                 path_2__1_2, reorder_path_host, nullptr,
                                 LEFT, grid_size, block_size, join_detail));

    tc_scc.fixpoint_loop();
```

### References
- [Getting Started on ThetaGPU](https://docs.alcf.anl.gov/theta-gpu/getting-started/)
- [Getting Started on Polaris](https://docs.alcf.anl.gov/polaris/getting-started/)
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)
- [CUDA - Pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)
