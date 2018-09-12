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
#include <curand_kernel.h>

// Kernel function to add the elements of two arrays
__global__
void cuda_fitness(int *data, int *results) {
    int individual = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int individual_start = individual * SIDE;
    int fitness = 0;
    for (int i = 0; i < SIDE; i++) {
        for (int j = i + 1; j < SIDE; j++) {
            if (data[individual_start + i] == data[individual_start + j] ||
                i - data[individual_start + i] == j - data[individual_start + j] ||
                i + data[individual_start + i] == j + data[individual_start + j]) {
                fitness++;
            }
        }
    }
    results[individual] = fitness;
}

__global__
void cuda_reproduce(int *data, int* data_next, int *results, int *results_idx, curandState *states) {
    int individual = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int individual_start = individual * SIDE;

    int parent1 = (int) (curand_uniform(&states[individual]) * TOURNAMENT_BEST);
    int parent2 = (int) (curand_uniform(&states[individual]) * TOURNAMENT_BEST);

    int parent1_idx = results_idx[parent1];
    int parent2_idx = results_idx[parent2];

    int parent1_start = parent1_idx * SIDE;
    int parent2_start = parent2_idx * SIDE;


    for (int i = 0; i < SIDE; i++) {
        if (curand_uniform(&states[individual]) < MUTATION_PROBABILITY) {
            data_next[individual_start + i] = (int) (curand_uniform(&states[individual]) * SIDE);
        } else {
            if (curand_uniform(&states[individual]) > 0.5) {
                data_next[individual_start + i] = data[parent1_start + i];
            } else {
                data_next[individual_start + i] = data[parent2_start + i];
            }
        }
    }
}

__global__
void setup_kernel(curandState *states)
{
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    curand_init(clock()*id, 0, 0, &states[id]);
}

__global__
void printCuda(int *array1, int *array2)
{
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    printf("%d - %d\n", array1[id], array2[id]);
}


struct RandGen {
    unsigned int clock;

    RandGen(unsigned int _clock) : clock(_clock) {}

    __device__
    float operator()(int idx) {
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
    thrust::device_vector<int> results_idx(POPULATION_SIZE);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t fitness_completed;
    cudaEventCreate (&fitness_completed);


    // Setups states for curand generation
    thrust::device_vector<curandState> curand_states(POPULATION_SIZE);
    setup_kernel<<<POPULATION_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(&curand_states[0])
    );

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(POPULATION_SIZE * SIDE),
        data.begin(),
        RandGen(static_cast<unsigned int>(clock()))
    );

    bool finished = false;
    thrust::host_vector<int> solution(SIDE);

    int generations = 0;
    int min_fitness[1];

    while (!finished) {
        generations++;

        cuda_fitness <<<POPULATION_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>> (
                thrust::raw_pointer_cast(&data[0]),
                thrust::raw_pointer_cast(&results[0])
        );

        thrust::sequence(thrust::cuda::par.on(stream1), results_idx.begin(), results_idx.end());
        thrust::sort_by_key(thrust::cuda::par.on(stream1), results.begin(), results.end(), results_idx.begin());

        cudaEventRecord (fitness_completed, stream1);

        cuda_reproduce <<<POPULATION_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>> (
            thrust::raw_pointer_cast(&data[0]),
            thrust::raw_pointer_cast(&data_next[0]),
            thrust::raw_pointer_cast(&results[0]),
            thrust::raw_pointer_cast(&results_idx[0]),
            thrust::raw_pointer_cast(&curand_states[0])
        );

        cudaEventSynchronize(fitness_completed);

        cudaMemcpyAsync(min_fitness, thrust::raw_pointer_cast(&results[0]), sizeof(int), cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream2);
        if (min_fitness[0] == 0) {
            int individual_start = results_idx[0] * SIDE;
            thrust::copy(data.begin() + individual_start, data.begin() + individual_start + SIDE, solution.begin());
            finished = true;
            break;
        } else {
            // Save fitness
        }

        data.swap(data_next);
    }

    std::cout << "Finished in " << generations << " generations" << std::endl;
    for (int i = 0; i < SIDE; i++) {
        std::cout << solution[i] << " ";
    }
    std::cout << std::endl;
}
