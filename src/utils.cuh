#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

typedef uint32_t id_t;
typedef float fp_t;

void clean_up_memory(std::vector<void*>& buffers);