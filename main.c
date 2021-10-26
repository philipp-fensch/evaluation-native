#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
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
    const int DIM = 3;
    
    double matrix_host[] = {
        2.0, 2.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 1.0, 1.0
    };

    double right_side_host[] = {
        2.0,
        4.0,
        5.0
    };

    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    TimeDiff diff;
    TimeDiff_start(&diff);

    double *matrix_device;
    double *right_side_device;
    int *pivoting_sequence_device;
    int *info_device;
    cudaMalloc((void**)&matrix_device, sizeof(double) * DIM * DIM);
    cudaMalloc((void**)&right_side_device, sizeof(double) * DIM);
    cudaMalloc((void**)&pivoting_sequence_device, sizeof(int) * DIM);
    cudaMalloc((void**)&info_device, sizeof(int));

    cudaMemcpy(matrix_device, matrix_host, sizeof(double) * DIM * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(right_side_device, right_side_host, sizeof(double) * DIM, cudaMemcpyHostToDevice);

    int workspace_size;
    double *workspace_device;
    cusolverDnDgetrf_bufferSize(cusolver_handle, DIM, DIM, matrix_device, DIM, &workspace_size);
    cudaMalloc((void**)&workspace_device, sizeof(double) * workspace_size);

    cusolverDnDgetrf(cusolver_handle, DIM, DIM, matrix_device, DIM, workspace_device, pivoting_sequence_device, info_device);
    cudaDeviceSynchronize();

    cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_T, DIM, 1, matrix_device, DIM, pivoting_sequence_device, right_side_device, DIM, info_device);
    cudaDeviceSynchronize();

    cudaMemcpy(right_side_host, right_side_device, DIM, cudaMemcpyDeviceToHost);
    
    TimeDiff_stop(&diff);
    printf("linear solver: %fms\n", TimeDiff_msec(&diff));
}

void matMulTest() {
    const int DIM = 3;

    double matrix_host[] = {
        2.0, 2.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 1.0, 1.0
    };

    TimeDiff diff;

    cublasHandle_t handle;
    cublasCreate(&handle);

    TimeDiff_start(&diff);
    double *device_matrix_A, *device_matrix_B, *d_C;
    cudaMalloc((void**)&device_matrix_A, DIM * DIM * sizeof(double));
    cudaMalloc((void**)&device_matrix_B, DIM * DIM * sizeof(double));

    cudaMemcpy(device_matrix_A, matrix_host, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1;
    const double beta = 0;
    cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, DIM, DIM, &alpha,
    device_matrix_A, DIM,
    device_matrix_A, DIM, &beta,
    device_matrix_B, DIM);

    cudaMemcpy(matrix_host, device_matrix_B, DIM * DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_A);
    cudaFree(device_matrix_B);

    TimeDiff_stop(&diff);
    printf("Matmul: %fµs\n", TimeDiff_usec(&diff));
}

int main() {
    memcpyTest();
    latencyTest();
    linearSolverTest();
    matMulTest();
    return 0;
}
