## Datasets
- Datasets are listed in [data folder](data).

## Dependencies
### Hardware
- The complete benchmark of the HIP-based experiments can be executed on an AMD MI250 GPU with a minimum of 64 GB GPU memory. The JLSE gpu_amd_mi250 node is a suitable choice.
- Partial benchmarks can be run on other AMD GPUs, but they may result in program termination for certain datasets due to limited GPU memory, leading to an instance of the out of memory error.
- AMD GPU MI250:
  - Supermicro AS-4124GQ-TNMI
  - 2x AMD EPYC 7713 64c (Milan) 64C/128T 2.0GHz 256M 225W
  - x4 AMD Instinct MI250 OAM 64GBx2 530W HBM2e PCIe Gen4
  - 512GB DDR4-3200
  - 1x EDR IB
- AMD GPU MI50:
  - Gigabyte G482-Z51
  - 2x 7742 64c Rome
  - 4x AMD MI50 32GB GPUs
  - Infinity Fabric
  - 256GB DDR-3200 RAM

### HIP (version 5.7 or later)
- Download and install the rocm 5.7 or later
### CMake 
- Download and install CMake(version 3.21 or later) from the CMake website: [https://cmake.org/download/](https://cmake.org/download/)
## Thrust
- need apply patch https://github.com/NVIDIA/thrust/pull/1832/files to fix integer overflow in `thrust::reduce`

## Run the program
- We present a HIP based Same Generation (SG), reachability, Context-Sensitive Program Analysis (CSPA) program that is optimized for sparse graphs.
- Build and run instructions are provided below:
```shell
cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S./ -B./build 
cd build
make
```
This will build the executables using the `hipcc` compiler.
- Each executable takes a single argument, which is the path to the input file containing the graph data. The input file should be in the following format:
```shell
./TC ../data/data_5.txt

# Benchmark SG
# fe_body
./SG ../data/data_163734.txt

# loc-Brightkite
./SG ../data/data_214078.txt

# fe_sphere
./SG ../data/data_49152.txt

# SF.cedge
./SG ../data/data_223001.txt

# CA-HepTh
./SG ../data/data_51971.txt

# ego-Facebook
./SG ../data/data_88234.txt

# Benchmark CSPA
# httpd
./CSPA ../data/dataset/httpd

# linux
./CSPA ../data/dataset/linux

# postgresql
./CSPA ../data/dataset/postgresql

# Benchmark Reachability
# com-dblp
./TC ../data/com-dblp.ungraph.txt

# fe_ocean
./TC ../data/data_409593.txt

# vsp_finan
./TC ../data/vsp_finan512_scagr7-2c_rlfddd.mtx

# p2p-Gnutella31
./TC ../data/data_147892.txt

# fe_body
./TC ../data/data_163734.txt

# SF.cedge
./TC ../data/data_223001.txt
```
### Run instructions for JLSE
- Run using Interactive node:
```shell
ssh <USERNAME>@login.jlse.anl.gov
nodelist
qsub -n 1 -q gpu_amd_mi250 -t 01:00:00 -I
module use /soft/modulefiles
module purge
module avail | grep rocm
module load rocm/5.7.0
module load cmake
module load gcc
hipcc --version
HIP version: 5.7.31921-d1770ee1b
AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.7.0 23352 d1e13c532a947d0cbfc94759c00dcf152294aa13)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /soft/compilers/rocm/rocm-5.7.0/llvm/bin

<USERNAME>@amdgpu01:~/gdlog/gdlog/build> module list
Currently Loaded Modulefiles:
 1) rocm/5.7.0   2) cmake/3.27.2   3) cuda/11.2.2   4) gcc/13.1.0 
<USERNAME>@amdgpu04:~> rocm-smi


========================= ROCm System Management Interface =========================
=================================== Concise Info ===================================
GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
0    39.0c           92.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
1    28.0c           N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
2    34.0c           91.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
3    36.0c           N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
4    33.0c           97.0W   800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
5    35.0c           N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
6    41.0c           100.0W  800Mhz  1600Mhz  0%   auto  560.0W    0%   0%    
7    37.0c           N/A     800Mhz  1600Mhz  0%   auto  0.0W      0%   0%    
====================================================================================
=============================== End of ROCm SMI Log ================================
WARNING: Unlocked monitor_devices lock; it should have already been unlocked.


rm -rf build
cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S./ -B./build 
cd build
make
# To select one GPU in a multiGPU system
export HIP_VISIBLE_DEVICES=1
./TC ../data/data_5.txt
```
- Transfer a local directory to JLSE:
```shell
scp -r /media/shovon/Codes/GithubCodes/gdlog <USERNAME>@login.jlse.anl.gov:/home/<USERNAME>/gdlog
scp -r /media/shovon/Codes/GithubCodes/gdlog/data/dataset/ <USERNAME>@login.jlse.anl.gov:/home/<USERNAME>/gdlog/gdlog/data/dataset/
```

### HIP Benchmarking result
- Benchmarking results can be found in [Benchmarking_Results.md](Benchmarking_Results.md)

### Run cuDF on Polaris
```shell
ssh <USERNAME>@polaris.alcf.anl.gov
qsub -I -l select=1 -l filesystems=home:grand:eagle -l walltime=1:00:00 -q debug -A dist_relational_alg
module purge
module load conda/2023-10-04
conda activate
pip install --extra-index-url https://pypi.nvidia.com cudf-cu11
python test/cuDF/sg.py

(2022-09-08/base) arsho::x3004c0s7b0n0 { ~/slog-gpu-backend/test/cuDF }-> python sg.py
| Dataset | Number of rows | SG size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| hipc | 5 | 4 | 3 | 0.016371 |
Error in fe_body. Message: std::bad_alloc: out_of_memory: CUDA error at: /__w/rmm/rmm/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
Error in loc-Brightkite. Message: std::bad_alloc: out_of_memory: CUDA error at: /__w/rmm/rmm/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
Error in fe_sphere. Message: std::bad_alloc: out_of_memory: CUDA error at: /__w/rmm/rmm/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
| CA-HepTh | 51971 | 74618689 | 9 | 21.241212 |
| Dataset | Number of rows | SG size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| ego-Facebook | 88234 | 15018986 | 13 | 19.074940 |
| wiki-Vote | 103689 | 5376338 | 4 | 2.603751 |
| luxembourg_osm | 119666 | 245221 | 326 | 2.215113 |
| cti | 48232 | 14503742 | 44 | 3.857438 |
| fe_ocean | 409593 | 65941441 | 77 | 45.979235 |
| wing | 121544 | 647999 | 8 | 0.204277 |
| delaunay_n16 | 196575 | 25994011 | 85 | 14.832548 |
Error in usroads. Message: std::bad_alloc: out_of_memory: CUDA error at: /__w/rmm/rmm/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
Error in p2p-Gnutella31. Message: std::bad_alloc: out_of_memory: CUDA error at: /__w/rmm/rmm/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
| p2p-Gnutella09 | 26013 | 62056583 | 14 | 13.705286 |
| p2p-Gnutella04 | 39994 | 116931333 | 18 | 48.947088 |
| cal.cedge | 21693 | 23519 | 58 | 0.259069 |
| TG.cedge | 23874 | 608090 | 54 | 0.719743 |
| OL.cedge | 7035 | 285431 | 56 | 0.385674 |
```


### References
- [Getting Started on Polaris](https://docs.alcf.anl.gov/polaris/getting-started/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)
