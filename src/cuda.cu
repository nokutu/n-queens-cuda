#include <iostream>
#include <math.h>
#include <curand_mtgp32_kernel.h>
#include <device_launch_parameters.h>
#include "constants.hpp"


#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/sort.h>

// Kernel function to add the elements of two arrays
__global__
void cuda_fitness(int *data, int *results) {
    int individual = threadIdx.x + blockIdx.x * (THREADS_PER_BLOCK / SIDE);
    int individual_start = individual * (SIDE + 1);
    for (int i = threadIdx.y; i < SIDE; i++) {
        for (int j = i + 1; j < SIDE; j++) {
            if (data[individual_start + i] == data[individual_start + j] ||
                i - data[individual_start + i] == j - data[individual_start + j] ||
                i + data[individual_start + i] == j + data[individual_start + j]) {

                results[individual]++;
            }
        }
    }
}

__global__
void cuda_reproduce(int *data, int *results, int *results_idx) {
    int individual = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int individual_start = individual * (SIDE + 1);

    // TODO
}

struct RandGen {
    unsigned int clock;

    RandGen(unsigned int _clock) : clock(_clock) {}

    __device__
    float operator()(unsigned int idx) {
        thrust::default_random_engine rng(clock);
        thrust::uniform_int_distribution<int> dist(0, SIDE);
        rng.discard(idx);
        return dist(rng);
    }
};

void run() {
    // Initialize data
    thrust::device_vector<int> data(POPULATION_SIZE * SIDE);
    thrust::device_vector<int> data_next(POPULATION_SIZE * SIDE);
    thrust::device_vector<int> results(POPULATION_SIZE);
    thrust::transform(
            data.begin(),
            data.end(),
            data.begin(),
            RandGen(static_cast<unsigned int>(clock()))
    );
    thrust::fill(data.begin(), data.end(), 0);

    bool finished = false;
    int solution[SIDE];

    int generations = 0;
    int min_fitness;

    // while (!finished) {
    cuda_fitness << < POPULATION_SIZE / (THREADS_PER_BLOCK / SIDE), dim3(THREADS_PER_BLOCK / SIDE, SIDE, 1) >> > (
            thrust::raw_pointer_cast(&data[0]),
            thrust::raw_pointer_cast(&results[0])
    );

    thrust::device_vector<int> results_idx(POPULATION_SIZE);
    thrust::sequence(results_idx.begin(), results_idx.end());
    thrust::sort_by_key(results.begin(), results.end(), results_idx.begin());

    for (int i = 0; i < 100; i++) {
        std::cout << results_idx[i] << " " << results[i] << std::endl;
    }
    //}

}
