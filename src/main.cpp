#include <iostream>
#include "cuda.hpp"
#include <chrono>
#include "constants.hpp"

using namespace std;
using namespace std::chrono;
int main() {

    uint64_t sum = 0;
    for (int i = 0; i < MEASURE_TRIES; i++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        run();
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>( t2 - t1 ).count();
        sum += duration;
    }
    

    cout << "Average: " << (sum / MEASURE_TRIES) / 1000 << " ms" << endl;

    return 0;
}

