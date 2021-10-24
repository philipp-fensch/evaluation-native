#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "time_diff.h"

void memcpyTest() {
    const size_t N_BYTES = 2 << 29; // 512 MiB

    void *memory_host = malloc(N_BYTES);
    void *memory_dev = NULL;
    cudaError_t error;
    TimeDiff diff;


    TimeDiff_start(&diff);
    error = cudaMalloc(&memory_dev, N_BYTES);
    cudaDeviceSynchronize();
    TimeDiff_stop(&diff);
    printf("cudaMalloc: %fms\n", TimeDiff_msec(&diff));

    TimeDiff_start(&diff);
    error = cudaMemcpy(memory_dev, memory_host, N_BYTES, cudaMemcpyHostToDevice);
    TimeDiff_stop(&diff);
    printf("cudaMemcpy: %fms\n", TimeDiff_msec(&diff));

    cudaFree(memory_dev);
    free(memory_host);
}

void latencyTest() {
    cudaError_t error;
    int count;
    TimeDiff diff;

    TimeDiff_start(&diff);
    error = cudaGetDeviceCount(&count);
    TimeDiff_stop(&diff);
    printf("cudaGetDeviceCount: %ldµs\n", TimeDiff_usec(&diff));
}

void linearSolverTest() {

}

void matMulTest() {

}

int main() {
    memcpyTest();
    latencyTest();
    return 0;
}
