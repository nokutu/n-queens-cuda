#include <iostream>
#include <random>
#include <cstring>
#include <climits>
#include "cuda.hpp"
#include "constants.hpp"

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> rand_real(0, 1);
    std::uniform_int_distribution<int> rand_side(0, SIDE);

    // Initialize data
    int* data = new int[POPULATION_SIZE * SIDE];
    int* data_next = new int[POPULATION_SIZE * SIDE];
    int* results = new int[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE * SIDE; i++) {
        data[i] = rand_side(gen);
    }

    bool finished = false;
    int solution[SIDE];

    int generations = 0;
    int min_fitness;

    while (!finished) {
        fitness(SIDE, POPULATION_SIZE, data, results);

        generations++;
        min_fitness = INT_MAX;

        std::vector<int> weights(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            int fitness = results[i];
            if (fitness < min_fitness) {
                min_fitness = fitness;
            }
            if (fitness == 0) {
                finished = true;
                std::memcpy(
                        data + i * SIDE * sizeof(int),
                        solution,
                        SIDE * sizeof(int)
                );
                break;
            }
            weights[i] = (SIDE*(SIDE-1)/2 - fitness) * (SIDE*(SIDE-1)/2 - fitness);
        }

        std::cout << "Generation: " << generations << ". Min fitness: " << min_fitness << std::endl;

        std::discrete_distribution<> d(weights.begin(), weights.end());

        for (int i = 0; i < POPULATION_SIZE; i++) {
            int individual1 = d(gen);
            int individual2 = d(gen);

            // Crossover & mutation
            int newIndividual[SIDE];
            for (int j = 0; j < SIDE; j++) {
                if (random() % 2 == 0) {
                    newIndividual[j] = data[individual1 * (SIDE + 1) + j];
                } else {
                    newIndividual[j] = data[individual2 * (SIDE + 1) + j];
                }

                if (rand_real(gen) < MUTATION_PROBABILITY) {
                    newIndividual[j] = rand_side(gen);
                }
            }

            std::memcpy(
                    data_next + i * SIDE * sizeof(int),
                    newIndividual,
                    SIDE * sizeof(int)
            );
        }

        std::swap(data, data_next);
    }

    clearCuda();
    return 0;
}

