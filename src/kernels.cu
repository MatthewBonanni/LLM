#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "utils.cuh"

using namespace nvcuda;

__device__ __host__ fp_t gelu(fp_t x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

template <uint32_t BLOCK_SIZE>
__global__ void embedding_kernel(
        const id_t* __restrict__ token_ids,
        const fp_t* __restrict__ wte,
        const fp_t* __restrict__ wpe,
        fp_t* __restrict__ embeddings,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset,
        uint32_t n_embd) {
    // Calculate thread ID
    const uint32_t idx_batch = blockIdx.x;
    const uint32_t idx_seq = blockIdx.y;
    const uint32_t tidx = threadIdx.x;

    // Check bounds
    if (idx_batch < batch_size &&
        idx_seq   < seq_length) {
        // Shared memory for token ID (one per block)
        __shared__ id_t token_id;
        if (tidx == 0) {
            token_id = token_ids[idx_batch * seq_length + idx_seq];
        }
        __syncthreads();
        
        // Calculate base offsets
        const uint64_t out_offset = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;
        const uint64_t wte_offset = (uint64_t)token_id * n_embd;
        const uint64_t wpe_offset = ((uint64_t)idx_seq + seq_offset) * n_embd;
        
        // Iterate over the embedding dimension in chunks of 4
        #pragma unroll
        for (uint32_t i = tidx * 4; i < n_embd; i += BLOCK_SIZE * 4) {
            // Load 4 elements from wte and wpe
            const float4 wte_vec = *reinterpret_cast<const float4*>(&wte[wte_offset + i]);
            const float4 wpe_vec = *reinterpret_cast<const float4*>(&wpe[wpe_offset + i]);
            
            // Sum the two vectors
            *reinterpret_cast<float4*>(&embeddings[out_offset + i]) = make_float4(
                wte_vec.x + wpe_vec.x,
                wte_vec.y + wpe_vec.y,
                wte_vec.z + wpe_vec.z,
                wte_vec.w + wpe_vec.w
            );
        }
    }
}

// Explicit instantiation
template __global__ void embedding_kernel<128>(
        const id_t* __restrict__ token_ids,
        const fp_t* __restrict__ wte,
        const fp_t* __restrict__ wpe,
        fp_t* __restrict__ embeddings,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset,
        uint32_t n_embd);

template <uint32_t BLOCK_SIZE, uint32_t WARPS_PER_BLOCK>
__global__ void layer_normalization_kernel(
        fp_t* __restrict__ input,
        const fp_t* __restrict__ gamma,
        const fp_t* __restrict__ beta,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    const uint32_t idx_batch = blockIdx.x;
    const uint32_t idx_seq = blockIdx.y;
    const uint32_t tidx = threadIdx.x;
    const uint32_t lane_id = tidx % WARP_SIZE;
    const uint32_t warp_id = tidx / WARP_SIZE;

    // Check bounds
    if (idx_batch < batch_size &&
        idx_seq   < seq_length) {
        // Get the starting index for the current token
        const uint64_t offset_input = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;
        
        // Shared memory for partial sums - organized by warp for efficient access
        __shared__ fp_t s_mean[WARPS_PER_BLOCK];
        __shared__ fp_t s_variance[WARPS_PER_BLOCK];
        
        // Local accumulators
        fp_t sum = 0.0f;
        fp_t sq_sum = 0.0f;
        
        // Calculate local sum and squared sum (with coalesced memory access)
        #pragma unroll
        for (uint32_t i = tidx * 4; i < n_embd; i += BLOCK_SIZE * 4) {
            float4 val4 = *reinterpret_cast<float4*>(&input[offset_input + i]);
            sum += val4.x + val4.y + val4.z + val4.w;
            sq_sum += val4.x * val4.x +
                    val4.y * val4.y +
                    val4.z * val4.z +
                    val4.w * val4.w;
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum    += __shfl_down_sync(0xffffffff, sum,    offset);
            sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
        }
        
        // First thread in each warp writes partial results
        if (lane_id == 0) {
            s_mean[warp_id] = sum;
            s_variance[warp_id] = sq_sum;
        }
        __syncthreads();
        
        // Final reduction across warps (done by first warp)
        if (warp_id == 0) {
            // Load 0 for lanes that would access out of bounds
            fp_t warp_sum = (lane_id < WARPS_PER_BLOCK) ? s_mean[lane_id] : 0.0f;
            fp_t warp_sq_sum = (lane_id < WARPS_PER_BLOCK) ? s_variance[lane_id] : 0.0f;
            
            // Warp-level reduction again
            #pragma unroll
            for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                warp_sq_sum += __shfl_down_sync(0xffffffff, warp_sq_sum, offset);
            }
            
            // First thread calculates final values
            if (lane_id == 0) {
                const fp_t inv_n = 1.0f / n_embd;
                s_mean[0] = warp_sum * inv_n;
                fp_t variance = fmaxf(warp_sq_sum * inv_n - s_mean[0] * s_mean[0], 0.0f);
                s_variance[0] = rsqrtf(variance + 1e-5f);  // inverse standard deviation
            }
        }
        __syncthreads();
        
        // Load final mean and inv_std
        const fp_t mean = s_mean[0];
        const fp_t inv_std = s_variance[0];
        
        // Normalize and scale - ensure coalesced memory access
        // Each thread handles multiple sequential elements for better instruction throughput
        #pragma unroll
        for (uint32_t i = tidx * 4; i < n_embd; i += BLOCK_SIZE * 4) {
            float4 input_vec = *reinterpret_cast<float4*>(&input[offset_input + i]);
            float4 gamma_vec = *reinterpret_cast<const float4*>(&gamma[i]);
            float4 beta_vec = *reinterpret_cast<const float4*>(&beta[i]);

            input_vec.x = (input_vec.x - mean) * inv_std * gamma_vec.x + beta_vec.x;
            input_vec.y = (input_vec.y - mean) * inv_std * gamma_vec.y + beta_vec.y;
            input_vec.z = (input_vec.z - mean) * inv_std * gamma_vec.z + beta_vec.z;
            input_vec.w = (input_vec.w - mean) * inv_std * gamma_vec.w + beta_vec.w;

            *reinterpret_cast<float4*>(&input[offset_input + i]) = input_vec;
        }
    }
}

// Explicit instantiation
template __global__ void layer_normalization_kernel<256, 8>(
        fp_t* __restrict__ input,
        const fp_t* __restrict__ gamma,
        const fp_t* __restrict__ beta,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd);

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
__global__ void q_projection_kernel(
        const fp_t* __restrict__ hidden_states,
        fp_t* __restrict__ q,
        const fp_t* __restrict__ w_q,
        const fp_t* __restrict__ b_q,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Warp index within the block
    const uint32_t thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;
    const uint32_t warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_m = warp_id / (BLOCK_M / WMMA_M);
    const uint32_t warp_n = warp_id % (BLOCK_M / WMMA_M);

    // Block-level tile starting positions, index into the output matrix
    const uint32_t block_tile_m = blockIdx.y * BLOCK_M;
    const uint32_t block_tile_n = blockIdx.x * BLOCK_N;

    // Warp-level tile starting positions, index into the output matrix
    const uint32_t warp_tile_m = block_tile_m + warp_m * WMMA_M;
    const uint32_t warp_tile_n = block_tile_n + warp_n * WMMA_N;

    // How many elements each thread will load into shared memory
    // from A (hidden_states) and B (w_q)
    const uint32_t A_elements_per_thread = (BLOCK_M * BLOCK_K + threads_per_block - 1) / threads_per_block;
    const uint32_t B_elements_per_thread = (BLOCK_K * BLOCK_N + threads_per_block - 1) / threads_per_block;
    const uint32_t C_elements_per_thread = (BLOCK_M * BLOCK_N + threads_per_block - 1) / threads_per_block;

    // Define shared memory for A and B tiles
    extern __shared__ half smem[];
    half* hidden_shared = &smem[0];
    half* w_q_shared = &smem[BLOCK_M * BLOCK_K];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragment with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over k dimension
    for (uint32_t block_tile_k = 0; block_tile_k < n_embd; block_tile_k += BLOCK_K) {
        // Load A (hidden_states) from global to shared memory
        for (uint32_t i = 0; i < A_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_K) {
                uint32_t m_idx = element_id / BLOCK_K;
                uint32_t k_idx = element_id % BLOCK_K;
                
                // Check bounds
                if ((block_tile_m + m_idx) < batch_size &&
                    (block_tile_k + k_idx) < n_embd) {
                    // Calculate global index for A (MxDxK, row-major)
                    // A[m][d_idx][k] -> m * (D * K) + d_idx * K + k
                    uint64_t global_idx = (block_tile_m + m_idx) * seq_length * n_embd +
                                          (seq_length - 1) * n_embd +
                                          (block_tile_k + k_idx);
                    hidden_shared[m_idx * BLOCK_K + k_idx] = __float2half(hidden_states[global_idx]);
                } else {
                    hidden_shared[m_idx * BLOCK_K + k_idx] = __float2half(0.0f);
                }
            }
        }

        // Load B (w_q) from global to shared memory
        for (uint32_t i = 0; i < B_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_K * BLOCK_N) {
                uint32_t k_idx = element_id % BLOCK_K;
                uint32_t n_idx = element_id / BLOCK_K;
                
                // Check bounds
                if ((block_tile_k + k_idx) < n_embd && (block_tile_n + n_idx) < n_embd) {
                    // Calculate global index for B (KxN, column-major)
                    // B[k][n] -> k + K * n
                    uint32_t global_idx = (block_tile_k + k_idx) + n_embd * (block_tile_n + n_idx);
                    w_q_shared[k_idx + BLOCK_K * n_idx] = __float2half(w_q[global_idx]);
                } else {
                    w_q_shared[k_idx + BLOCK_K * n_idx] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Loop over the shared tile with tensor core operations
        for (uint32_t k_warp = 0; k_warp < BLOCK_K; k_warp += WMMA_K) {
            // Load A and B from shared memory into fragments
            uint32_t a_row = warp_m * WMMA_M;
            uint32_t a_col = k_warp;
            uint32_t b_row = k_warp;
            uint32_t b_col = warp_n * WMMA_N;
            
            // Load matrix fragments from shared memory
            wmma::load_matrix_sync(a_frag, &hidden_shared[a_row * BLOCK_K + a_col], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &w_q_shared[b_row + BLOCK_K * b_col], BLOCK_K);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Ensure all warps have finished using shared memory
        __syncthreads();
    }

    // Store the output
    if (warp_tile_m < batch_size && warp_tile_n < n_embd) {
        wmma::store_matrix_sync(
            &q[warp_tile_m * n_embd + warp_tile_n],
            c_frag,
            n_embd,
            wmma::mem_row_major);
        
        // Add bias
        for (uint32_t i = 0; i < C_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_N) {
                uint32_t m_idx = element_id / BLOCK_N;
                uint32_t n_idx = element_id % BLOCK_N;

                // Check bounds
                if ((warp_tile_m + m_idx) < batch_size &&
                    (warp_tile_n + n_idx) < n_embd) {
                    // Add bias
                    q[(warp_tile_m + m_idx) * n_embd +
                      (warp_tile_n + n_idx)] += b_q[warp_tile_n + n_idx];
                }
            }
        }
    }
}

// Explicit instantiation
template __global__ void q_projection_kernel<32, 32, 32>(
        const fp_t* __restrict__ hidden_states,
        fp_t* __restrict__ q,
        const fp_t* __restrict__ w_q,
        const fp_t* __restrict__ b_q,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd);

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
__global__ void kv_projection_kernel(
        const fp_t* __restrict__ hidden_states,
        half* __restrict__ kv,
        const fp_t* __restrict__ w_kv,
        const fp_t* __restrict__ b_kv,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset,
        uint32_t n_embd) {
    // Warp index within the block
    const uint32_t thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;
    const uint32_t warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_m = warp_id / (BLOCK_M / WMMA_M);
    const uint32_t warp_n = warp_id % (BLOCK_M / WMMA_M);
    const uint32_t idx_batch = blockIdx.z * blockDim.z + threadIdx.z;

    // Block-level tile starting positions, index into the output matrix
    const uint32_t block_tile_m = blockIdx.y * BLOCK_M;
    const uint32_t block_tile_n = blockIdx.x * BLOCK_N;

    // Warp-level tile starting positions, index into the output matrix
    const uint32_t warp_tile_m = block_tile_m + warp_m * WMMA_M;
    const uint32_t warp_tile_n = block_tile_n + warp_n * WMMA_N;

    // How many elements each thread will load into shared memory
    // from A (hidden_states) and B (w_kv)
    // and write to C (kv)
    const uint32_t A_elements_per_thread = (BLOCK_M * BLOCK_K + threads_per_block - 1) / threads_per_block;
    const uint32_t B_elements_per_thread = (BLOCK_K * BLOCK_N + threads_per_block - 1) / threads_per_block;
    const uint32_t C_elements_per_thread = (BLOCK_M * BLOCK_N + threads_per_block - 1) / threads_per_block;

    // Define shared memory for A and B tiles
    extern __shared__ half smem[];
    half* hidden_shared = &smem[0];
    half* w_kv_shared = &smem[BLOCK_M * BLOCK_K];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragment with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    __syncthreads();

    // Loop over k dimension
    for (uint32_t block_tile_k = 0; block_tile_k < n_embd; block_tile_k += BLOCK_K) {
        // Load A (hidden_states) from global to shared memory
        for (uint32_t i = 0; i < A_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_K) {
                uint32_t m_idx = element_id / BLOCK_K;
                uint32_t k_idx = element_id % BLOCK_K;
                
                // Check bounds
                if ((block_tile_m + m_idx + seq_offset) < seq_length &&
                    (block_tile_k + k_idx) < n_embd) {
                    // Calculate global index for A (DxMxK, row-major)
                    // A[idx_batch][m][k] -> idx_batch * (M * K) + m * (K) + k
                    uint64_t global_idx = idx_batch * seq_length * n_embd +
                                          (block_tile_m + m_idx + seq_offset) * n_embd +
                                          (block_tile_k + k_idx);
                    hidden_shared[m_idx * BLOCK_K + k_idx] = __float2half(hidden_states[global_idx]);
                } else {
                    hidden_shared[m_idx * BLOCK_K + k_idx] = __float2half(0.0f);
                }
            }
        }

        // Load B (w_kv) from global to shared memory
        for (uint32_t i = 0; i < B_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_K * BLOCK_N) {
                uint32_t k_idx = element_id % BLOCK_K;
                uint32_t n_idx = element_id / BLOCK_K;
                
                // Check bounds
                if ((block_tile_k + k_idx) < n_embd && (block_tile_n + n_idx) < n_embd) {
                    // Calculate global index for B (KxN, column-major)
                    // B[k][n] -> k + N * n
                    uint32_t global_idx = (block_tile_k + k_idx) + n_embd * (block_tile_n + n_idx);
                    w_kv_shared[k_idx + BLOCK_N * n_idx] = __float2half(w_kv[global_idx]);
                } else {
                    w_kv_shared[k_idx + BLOCK_N * n_idx] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Loop over the shared tile with tensor core operations
        for (uint32_t k_warp = 0; k_warp < BLOCK_K; k_warp += WMMA_K) {
            // Load A and B from shared memory into fragments
            uint32_t a_row = warp_m * WMMA_M;
            uint32_t a_col = k_warp;
            uint32_t b_row = k_warp;
            uint32_t b_col = warp_n * WMMA_N;
            
            // Load matrix fragments from shared memory
            wmma::load_matrix_sync(a_frag, &hidden_shared[a_row * BLOCK_K + a_col], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &w_kv_shared[b_row + BLOCK_N * b_col], BLOCK_N);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Ensure all warps have finished using shared memory
        __syncthreads();
    }

    // Store the output
    if ((warp_tile_m + seq_offset) < seq_length &&
        warp_tile_n < n_embd) {
        // Store the output in shared memory
        float* output = reinterpret_cast<float*>(smem);
        wmma::store_matrix_sync(output, c_frag, WMMA_N, wmma::mem_row_major);

        // Add bias and transfer the output to KV
        for (uint32_t i = 0; i < C_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_N) {
                uint32_t m_idx = element_id / BLOCK_N;
                uint32_t n_idx = element_id % BLOCK_N;

                // Check bounds
                if ((warp_tile_m + m_idx + seq_offset) < seq_length &&
                    (warp_tile_n + n_idx) < n_embd) {
                    // Store the output in kv
                    kv[idx_batch * seq_length * n_embd +
                       (warp_tile_m + m_idx + seq_offset) * n_embd +
                       (warp_tile_n + n_idx)] = __float2half(output[m_idx * BLOCK_N + n_idx]) +
                                                __float2half(b_kv[warp_tile_n + n_idx]);
                }
            }
        }
    }
}

// Explicit instantiation
template __global__ void kv_projection_kernel<32, 32, 32>(
        const fp_t* __restrict__ hidden_states,
        half* __restrict__ kv,
        const fp_t* __restrict__ w_kv,
        const fp_t* __restrict__ b_kv,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset,
        uint32_t n_embd);

// __global__ void multi_head_attention_kernel(
//         const fp_t* __restrict__ q,
//         const half* __restrict__ k,
//         const half* __restrict__ v,
//         fp_t* __restrict__ output,
//         uint32_t batch_size,
//         uint32_t seq_length,
//         uint32_t seq_offset,
//         uint32_t n_head,
//         uint32_t n_embd) {
//     // Calculate indices
//     const uint32_t idx_head = blockIdx.y;
//     const uint32_t idx_seq = threadIdx.x;
//     const uint32_t idx_batch = blockIdx.z * blockDim.z + threadIdx.z;
//     const uint32_t lane_id = threadIdx.x % WARP_SIZE;
//     const uint32_t thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//     const uint32_t warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;

//     const uint32_t warp_m = warp_id / (BLOCK_M / WMMA_M);
//     const uint32_t warp_n = warp_id % (BLOCK_M / WMMA_M);

//     // Tensor Core matrices
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
//     // Compute head-specific dimensions
//     uint32_t d_k = n_embd / n_head;
//     float scale = rsqrtf(d_k);
    
//     // Compute attention scores using Tensor Cores
//     wmma::fill_fragment(c_frag, 0.0f);
    
//     // Parallel dot product computation using Tensor Cores
//     for (uint32_t k_block = 0; k_block < d_k; k_block += WMMA_K) {
//         // Load Q and K fragments
//         wmma::load_matrix_sync(
//             a_frag, 
//             reinterpret_cast<const __half*>(q + idx_batch * seq_length * n_embd + 
//                                             idx_seq * n_embd + 
//                                             idx_head * d_k + 
//                                             k_block), 
//             n_embd);
        
//         wmma::load_matrix_sync(
//             b_frag, 
//             reinterpret_cast<const __half*>(k + idx_batch * (seq_length + seq_offset) * n_embd + 
//                                             idx_seq * n_embd + 
//                                             idx_head * d_k + 
//                                             k_block), 
//             n_embd);
        
//         // Perform matrix multiplication
//         wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//     }
    
//     // Softmax computation with causal masking
//     float local_score = (idx_seq <= blockIdx.z) ? c_frag.x[0] * scale : -INFINITY;
    
//     // Parallel reduction for max and sum
//     __shared__ float s_max, s_sum;
//     float thread_max = local_score;
//     float thread_exp = (idx_seq <= blockIdx.z) ? expf(local_score) : 0.0f;
    
//     // Collaborative max reduction

//     // Reduction within the warp
//     for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//         thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
//     }

//     // First thread in the warp writes the result
//     if (lane_id == 0) {
//         s_max = thread_max;
//     }
//     __syncthreads();

//     // Final reduction across warps (done by first warp)
//     if (warp_id == 0) {
//         // Load 0 for lanes that would access out of bounds
//         float warp_max = (lane_id < WARPS_PER_BLOCK) ? s_max : -INFINITY;
        
//         // Warp-level reduction again
//         for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//             warp_max = max(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
//         }
        
//         // First thread calculates final values
//         if (lane_id == 0) {
//             s_max = warp_max;
//         }
//     }
//     __syncthreads();
    
//     // Exponential and sum reduction
//     float exp_scaled = expf(local_score - s_max);
//     float thread_sum = exp_scaled;

//     // Collaborative sum reduction

//     // Reduction within the warp
//     for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//         thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
//     }

//     // First thread in the warp writes the result
//     if (lane_id == 0) {
//         s_sum = thread_sum;
//     }
//     __syncthreads();

//     // Final reduction across warps (done by first warp)
//     if (warp_id == 0) {
//         // Load 0 for lanes that would access out of bounds
//         float warp_sum = (lane_id < WARPS_PER_BLOCK) ? s_sum : 0.0f;
        
//         // Warp-level reduction again
//         for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//             warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
//         }
        
//         // First thread calculates final values
//         if (lane_id == 0) {
//             s_sum = warp_sum;
//         }
//     }
//     __syncthreads();

//     // Final output computation with values
//     float attention_weight = (idx_seq <= blockIdx.z) ? exp_scaled / s_sum : 0.0f;
    
//     // Compute weighted sum of values using Tensor Cores
//     wmma::fill_fragment(a_frag, attention_weight);
//     wmma::fill_fragment(c_frag, 0.0f);
    
//     for (uint32_t k_block = 0; k_block < d_k; k_block += WMMA_K) {
//         // Load V fragment
//         wmma::load_matrix_sync(
//             b_frag,
//             reinterpret_cast<const __half*>(v + idx_batch * (seq_length + seq_offset) * n_embd + 
//                                             idx_seq * n_embd + 
//                                             idx_head * d_k + 
//                                             k_block), 
//             n_embd);
        
//         // Perform weighted value accumulation
//         wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//     }
    
//     // Store result
//     output[idx_batch * seq_length * n_embd +
//            idx_seq * n_embd +
//            idx_head * d_k] = c_frag.x[0];
// }

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
__global__ void final_projection_kernel(
        const fp_t* __restrict__ input,
        fp_t* __restrict__ output,
        const fp_t* __restrict__ w_proj,
        const fp_t* __restrict__ b_proj,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Warp index within the block
    const uint32_t thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;
    const uint32_t warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_m = warp_id / (BLOCK_M / WMMA_M);
    const uint32_t warp_n = warp_id % (BLOCK_M / WMMA_M);
    const uint32_t idx_batch = blockIdx.z * blockDim.z + threadIdx.z;

    // Block-level tile starting positions, index into the output matrix
    const uint32_t block_tile_m = blockIdx.y * BLOCK_M;
    const uint32_t block_tile_n = blockIdx.x * BLOCK_N;

    // Warp-level tile starting positions, index into the output matrix
    const uint32_t warp_tile_m = block_tile_m + warp_m * WMMA_M;
    const uint32_t warp_tile_n = block_tile_n + warp_n * WMMA_N;

    // How many elements each thread will load into shared memory
    // from A (input) and B (w_proj)
    // and write to C (output)
    const uint32_t A_elements_per_thread = (BLOCK_M * BLOCK_K + threads_per_block - 1) / threads_per_block;
    const uint32_t B_elements_per_thread = (BLOCK_K * BLOCK_N + threads_per_block - 1) / threads_per_block;
    const uint32_t C_elements_per_thread = (BLOCK_M * BLOCK_N + threads_per_block - 1) / threads_per_block;

    // Define shared memory for A and B tiles
    extern __shared__ half smem[];
    half* input_shared = &smem[0];
    half* w_proj_shared = &smem[BLOCK_M * BLOCK_K];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragment with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    __syncthreads();

    // Loop over k dimension
    for (uint32_t block_tile_k = 0; block_tile_k < n_embd; block_tile_k += BLOCK_K) {
        // Load A (input) from global to shared memory
        for (uint32_t i = 0; i < A_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_K) {
                uint32_t m_idx = element_id / BLOCK_K;
                uint32_t k_idx = element_id % BLOCK_K;
                
                // Check bounds
                if ((block_tile_m + m_idx) < seq_length &&
                    (block_tile_k + k_idx) < n_embd) {
                    // Calculate global index for A (DxMxK, row-major)
                    // A[idx_batch][m][k] -> idx_batch * (M * K) + m * (K) + k
                    uint64_t global_idx = idx_batch * seq_length * n_embd +
                                          (block_tile_m + m_idx) * n_embd +
                                          (block_tile_k + k_idx);
                    input_shared[m_idx * BLOCK_K + k_idx] = __float2half(input[global_idx]);
                } else {
                    input_shared[m_idx * BLOCK_K + k_idx] = __float2half(0.0f);
                }
            }
        }

        // Load B (w_proj) from global to shared memory
        for (uint32_t i = 0; i < B_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_K * BLOCK_N) {
                uint32_t k_idx = element_id % BLOCK_K;
                uint32_t n_idx = element_id / BLOCK_K;
                
                // Check bounds
                if ((block_tile_k + k_idx) < n_embd && (block_tile_n + n_idx) < n_embd) {
                    // Calculate global index for B (KxN, column-major)
                    // B[k][n] -> k + N * n
                    uint32_t global_idx = (block_tile_k + k_idx) + n_embd * (block_tile_n + n_idx);
                    w_proj_shared[k_idx + BLOCK_N * n_idx] = __float2half(w_proj[global_idx]);
                } else {
                    w_proj_shared[k_idx + BLOCK_N * n_idx] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Loop over the shared tile with tensor core operations
        for (uint32_t k_warp = 0; k_warp < BLOCK_K; k_warp += WMMA_K) {
            // Load A and B from shared memory into fragments
            uint32_t a_row = warp_m * WMMA_M;
            uint32_t a_col = k_warp;
            uint32_t b_row = k_warp;
            uint32_t b_col = warp_n * WMMA_N;
            
            // Load matrix fragments from shared memory
            wmma::load_matrix_sync(a_frag, &input_shared[a_row * BLOCK_K + a_col], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &w_proj_shared[b_row + BLOCK_N * b_col], BLOCK_N);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Ensure all warps have finished using shared memory
        __syncthreads();
    }

    // Store the output
    if (warp_tile_m < seq_length && warp_tile_n < n_embd) {
        wmma::store_matrix_sync(
            &output[warp_tile_m * n_embd + warp_tile_n],
            c_frag,
            n_embd,
            wmma::mem_row_major);
        
        // Add bias
        for (uint32_t i = 0; i < C_elements_per_thread; i++) {
            uint32_t element_id = thread_id + i * threads_per_block;
            if (element_id < BLOCK_M * BLOCK_N) {
                uint32_t m_idx = element_id / BLOCK_N;
                uint32_t n_idx = element_id % BLOCK_N;

                // Check bounds
                if ((warp_tile_m + m_idx) < seq_length &&
                    (warp_tile_n + n_idx) < n_embd) {
                    // Add bias
                    output[(warp_tile_m + m_idx) * n_embd +
                           (warp_tile_n + n_idx)] += b_proj[warp_tile_n + n_idx];
                }
            }
        }
    }
}

// Explicit instantiation
template __global__ void final_projection_kernel<32, 32, 32>(
        const fp_t* __restrict__ input,
        fp_t* __restrict__ output,
        const fp_t* __restrict__ w_proj,
        const fp_t* __restrict__ b_proj,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd);

template <uint32_t BLOCK_SIZE>
__global__ void add_residual_kernel(
        const fp_t* __restrict__ input,
        const fp_t* __restrict__ residual,
        fp_t* __restrict__ output,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    const uint32_t idx_batch = blockIdx.x;
    const uint32_t idx_seq = blockIdx.y;
    const uint32_t tidx = threadIdx.x;

    // Check bounds
    if (idx_batch < batch_size &&
        idx_seq   < seq_length) {
        // Get the starting index for the current token
        const uint64_t offset = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

        // Iterate over the embedding dimension in chunks of 4
        #pragma unroll
        for (uint32_t i = tidx * 4; i < n_embd; i += BLOCK_SIZE * 4) {
            // Load 4 elements from input and residual
            float4 input_vec = *reinterpret_cast<const float4*>(&input[offset + i]);
            float4 residual_vec = *reinterpret_cast<const float4*>(&residual[offset + i]);

            // Add residual
            *reinterpret_cast<float4*>(&output[offset + i]) = make_float4(
                input_vec.x + residual_vec.x,
                input_vec.y + residual_vec.y,
                input_vec.z + residual_vec.z,
                input_vec.w + residual_vec.w
            );
        }
    }
}

// Explicit instantiation
template __global__ void add_residual_kernel<128>(
        const fp_t* __restrict__ input,
        const fp_t* __restrict__ residual,
        fp_t* __restrict__ output,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd);

__global__ void mlp_kernel(
        const fp_t* __restrict__ input,
        fp_t* __restrict__ output,
        const fp_t* __restrict__ w_fc,
        const fp_t* __restrict__ b_fc, 
        const fp_t* __restrict__ w_proj,
        const fp_t* __restrict__ b_proj,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch < batch_size &&
        idx_seq   < seq_length) {
        // Intermediate register
        uint32_t intermediate_size = 4 * n_embd;
        fp_t intermediate[INTERMEDIATE_SIZE];

        // Get the starting index for the current token
        uint64_t offset_input = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

        // Compute feedforward layer
        for (uint32_t i = 0; i < intermediate_size; i++) {
            fp_t val = b_fc[i];
            for (uint32_t j = 0; j < n_embd; j++) {
                val += input[offset_input + j] * w_fc[j * intermediate_size + i];
            }
            intermediate[i] = gelu(val);
        }

        // Compute projection back to hidden size
        for (uint32_t i = 0; i < n_embd; i++) {
            fp_t val = b_proj[i];
            for (uint32_t j = 0; j < intermediate_size; j++) {
                val += intermediate[j] * w_proj[j * n_embd + i];
            }
            output[offset_input + i] = val;
        }
    }
}

template <uint32_t BLOCK_SIZE, uint32_t WARPS_PER_BLOCK>
__global__ void lm_head_kernel(
        const fp_t* __restrict__ hidden_state,
        fp_t* __restrict__ logits,
        const fp_t* __restrict__ weights,
        const fp_t* __restrict__ biases,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_vocab,
        uint32_t n_embd) {
    // Calculate thread ID
    const uint32_t idx_batch = blockIdx.x;
    const uint32_t idx_vocab = blockIdx.y;
    const uint32_t tidx = threadIdx.x;
    const uint32_t lane_id = tidx % WARP_SIZE;
    const uint32_t warp_id = tidx / WARP_SIZE;
    

    // Check bounds
    if (idx_batch < batch_size &&
        idx_vocab < n_vocab) {
        // Calculate output index
        uint64_t idx_out = ((uint64_t)idx_batch * n_vocab + idx_vocab);

        // Get the starting index for the current token
        uint64_t offset_input =  ((uint64_t)idx_batch * seq_length + (seq_length - 1)) * n_embd;
        uint64_t offset_weights = (uint64_t)idx_vocab * n_embd;

        // Shared memory for partial sums
        __shared__ float s_sums[WARPS_PER_BLOCK];

        // Local accumulator
        fp_t sum = biases ? biases[idx_vocab] : 0.0f;

        // Calculate local sum (with coalesced memory access)
        #pragma unroll
        for (uint32_t i = tidx * 4; i < n_embd; i += BLOCK_SIZE * 4) {
            float4 hidden_vec = *reinterpret_cast<const float4*>(&hidden_state[offset_input + i]);
            float4 weight_vec = *reinterpret_cast<const float4*>(&weights[offset_weights + i]);

            sum += hidden_vec.x * weight_vec.x +
                hidden_vec.y * weight_vec.y +
                hidden_vec.z * weight_vec.z +
                hidden_vec.w * weight_vec.w;
        }

        // Warp-level reduction
        #pragma unroll
        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // First thread in the warp writes the result
        if (lane_id == 0) {
            s_sums[warp_id] = sum;
        }
        __syncthreads();

        // Final reduction across warps (done by first warp)
        if (warp_id == 0) {
            // Load 0 for lanes that would access out of bounds
            float warp_sum = (lane_id < WARPS_PER_BLOCK) ? s_sums[lane_id] : 0.0f;
            
            // Warp-level reduction again
            for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }
            
            // First thread calculates final values
            if (lane_id == 0) {
                logits[idx_out] = warp_sum;
            }
        }
    }
}

// Explicit instantiation
template __global__ void lm_head_kernel<256, 8>(
        const fp_t* __restrict__ hidden_state,
        fp_t* __restrict__ logits,
        const fp_t* __restrict__ weights,
        const fp_t* __restrict__ biases,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_vocab,
        uint32_t n_embd);