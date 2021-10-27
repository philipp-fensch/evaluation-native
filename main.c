#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "time_diff.h"
#include "matrix.h"

#define DIM 5000
#define ITERATIONS 10

cublasHandle_t handle;
cusolverDnHandle_t cusolver_handle;

void memcpyTest() {
    const size_t N_BYTES = 2 << 29; // 512 MiB

    void *memory_host = malloc(N_BYTES);
    void *memory_dev = NULL;
    cudaError_t error;
    TimeDiff diff[ITERATIONS];


    error = cudaMalloc(&memory_dev, N_BYTES);

    for(int i = 0; i < ITERATIONS; i++) {
        TimeDiff_start(&diff[i]);
        error = cudaMemcpy(memory_dev, memory_host, N_BYTES, cudaMemcpyHostToDevice);
        TimeDiff_stop(&diff[i]);
        printf("cudaMemcpy: %.2fms\n", TimeDiff_msec(&diff[i]));
    }
    printf("cudaMemcpy Average: %.2fms\n\n", average_msec(diff, ITERATIONS));

    cudaFree(memory_dev);
    free(memory_host);
}

void latencyTest() {
    cudaError_t error;
    int count;
    TimeDiff diff[ITERATIONS];

    cudaDeviceSynchronize();
    for(int i = 0; i < ITERATIONS; i++) {
        TimeDiff_start(&diff[i]);
        error = cudaGetDeviceCount(&count);
        TimeDiff_stop(&diff[i]);
        printf("cudaGetDeviceCount: %.2fµs\n", TimeDiff_usec(&diff[i]));
    }
    printf("cudaGetDeviceCount Average: %.2fµs\n\n", average_usec(diff, ITERATIONS));
}

void linearSolverTest() {
    double *right_side_host = read_vector_from_file(DIM, "vector.txt");
    double *matrix_host = read_matrix_from_file(DIM, "matrix.txt");
    double *solution = malloc(DIM * sizeof(double));

    TimeDiff diff[ITERATIONS];

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

    // LU
    cusolverDnDgetrf(cusolver_handle, DIM, DIM, matrix_device, DIM, workspace_device, pivoting_sequence_device, info_device);
    cudaDeviceSynchronize();

    // Solve
    for(int i = 0; i < ITERATIONS; i++) {
        // Copy right side to device
        cudaMemcpy(right_side_device, right_side_host, sizeof(double) * DIM, cudaMemcpyHostToDevice);

        //
        TimeDiff_start(&diff[i]);
        cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_T, DIM, 1, matrix_device, DIM, pivoting_sequence_device, right_side_device, DIM, info_device);
        cudaDeviceSynchronize();
        TimeDiff_stop(&diff[i]);
        printf("linear solver: %.2fms\n", TimeDiff_msec(&diff[i]));
    }

    printf("linear solver Average: %.2fms\n\n", average_msec(diff, ITERATIONS));

    cudaMemcpy(solution, right_side_device, DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(matrix_device);
    cudaFree(right_side_device);
    cudaFree(pivoting_sequence_device);
    cudaFree(info_device);

    free(matrix_host);
    free(right_side_host);
    free(solution);
}

void matMulTest() {
    double *matrix_host = read_matrix_from_file(DIM, "matrix.txt");
    double *solution = malloc(DIM * DIM * sizeof(double));

    TimeDiff diff[ITERATIONS];
    double *device_matrix_A, *device_matrix_B, *d_C;
    cudaMalloc((void**)&device_matrix_A, DIM * DIM * sizeof(double));
    cudaMalloc((void**)&device_matrix_B, DIM * DIM * sizeof(double));

    cudaMemcpy(device_matrix_A, matrix_host, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);

    const double alpha = 1;
    const double beta = 0;

    // "Warming up"
    cublasStatus_t status = cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, DIM, DIM, &alpha,
            device_matrix_A, DIM,
            device_matrix_A, DIM, &beta,
            device_matrix_B, DIM);
    cudaDeviceSynchronize();

    // bench
    for(int i = 0; i < ITERATIONS; i++) {
        TimeDiff_start(&diff[i]);
        cublasStatus_t status = cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, DIM, DIM, &alpha,
            device_matrix_A, DIM,
            device_matrix_A, DIM, &beta,
            device_matrix_B, DIM);
        cudaDeviceSynchronize();
        TimeDiff_stop(&diff[i]);
        printf("Matmul: %.2fms\n", TimeDiff_msec(&diff[i]));
    }

    printf("Matmul Average: %.2fms\n\n", average_msec(diff, ITERATIONS));

    cudaMemcpy(solution, device_matrix_B, DIM * DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_A);
    cudaFree(device_matrix_B);
}

int main() {
    // Init
    cublasCreate(&handle);
    cusolverDnCreate(&cusolver_handle);

    memcpyTest();
    latencyTest();
    linearSolverTest();
    matMulTest();

    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(handle);
    return 0;
}
