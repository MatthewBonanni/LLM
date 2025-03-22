#include "utils.cuh"

#include <cuda_runtime.h>

void clean_up_memory(std::vector<void*>& buffers) {
    for (void*& buffer : buffers) {
        if (buffer != nullptr) {
            CHECK_CUDA(cudaFree(buffer));
        }
        buffer = nullptr;
    }
}