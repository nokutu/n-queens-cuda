cmake_minimum_required(VERSION 3.0)
project(n_queens_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

add_executable(n_queens_cuda ${cpu_source_files} ${gpu_source_files})