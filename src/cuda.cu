#include <iostream>
#include <math.h>
#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>
#include "constants.hpp"

// Kernel function to add the elements of two arrays
__global__
void cuda_fitness(int side, int population, int *data, int *results) {
    int individual = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int individual_start = individual * (side + 1);
    for (int i = 0; i < side; i++) {
        for (int j = i; j < side; j++) {
            if (i != j) {
                if (data[individual_start + i] == data[individual_start + j]) {

                } else if() {
                    
                }
            }
        }
    }
}

int *data_gpu;
int *results_gpu;
bool inited = false;

void fitness(int side, int population, int *data, int* results) {

    if (!inited) {
        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMalloc(&data_gpu, (side) * population * sizeof(int));
        cudaMalloc(&results_gpu, population * sizeof(int));
        inited = true;
    }

    // Copy input data to array on GPU.
    cudaMemcpy(data, data_gpu, (side + 1) * population * sizeof(int), cudaMemcpyHostToDevice);

    printf("%d", POPULATION_SIZE / THREADS_PER_BLOCK);
    // Run kernel on 1M elements on the GPU
    cuda_fitness <<<POPULATION_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> (side, population, data_gpu, results_gpu);

    // Wait for GPU to clearCuda before accessing on host
    cudaDeviceSynchronize();

    // Copy results
    cudaMemcpy(results_gpu, results, population * sizeof(int), cudaMemcpyDeviceToHost);
}

void clearCuda() {
    // Free memory
    cudaFree(data_gpu);
    cudaFree(results_gpu);
}