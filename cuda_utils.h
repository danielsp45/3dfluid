#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define CUDA(...) { \
    __VA_ARGS__; \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#endif //CUDA_UTILS_H
