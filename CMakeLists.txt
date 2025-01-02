cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fluid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(fluid_sim
        main.cu
        EventManager.cpp
        fluid_solver.cu
        fluid_solver.h
        cuda_utils.h
)

set_target_properties(fluid_sim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
