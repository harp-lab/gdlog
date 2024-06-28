
#pragma once
// #include <hip_runtime.h>

struct KernelTimer {
    hipEvent_t start;
    hipEvent_t stop;

    KernelTimer() {
        hipEventCreate(&start);
        hipEventCreate(&stop);
    }

    ~KernelTimer() {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    void start_timer() { hipEventRecord(start, 0); }

    void stop_timer() { hipEventRecord(stop, 0); }

    float get_spent_time() {
        float elapsed;
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0;
        return elapsed;
    }
};

struct Output {
    int block_size;
    int grid_size;
    long int input_rows;
    long int hashtable_rows;
    double load_factor;
    double initialization_time;
    double memory_clear_time;
    double read_time;
    double reverse_time;
    double hashtable_build_time;
    long int hashtable_build_rate;
    double join_time;
    double projection_time;
    double deduplication_time;
    double union_time;
    double total_time;
    const char *dataset_name;
};
