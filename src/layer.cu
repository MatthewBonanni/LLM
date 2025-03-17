#include "layer.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

#include "utils.cuh"
#include "io.cuh"

Layer::Layer(int n_embd, int n_head) : 
        n_embd(n_embd),
        n_head(n_head),
        d_attn_c_attn_w_0(nullptr),
        d_attn_c_attn_b_0(nullptr),
        d_attn_c_proj_w_0(nullptr),
        d_attn_c_proj_b_0(nullptr),
        d_ln_1_b_0(nullptr),
        d_ln_1_g_0(nullptr),
        d_ln_2_b_0(nullptr),
        d_ln_2_g_0(nullptr),
        d_mlp_c_fc_w_0(nullptr),
        d_mlp_c_fc_b_0(nullptr),
        d_mlp_c_proj_w_0(nullptr),
        d_mlp_c_proj_b_0(nullptr) {
    // Allocate memory on host
    h_attn_c_attn_w_0.resize(n_embd * 3 * n_embd);
    h_attn_c_attn_b_0.resize(3 * n_embd);
    h_attn_c_proj_w_0.resize(n_embd * n_embd);
    h_attn_c_proj_b_0.resize(n_embd);
    h_ln_1_b_0.resize(n_embd);
    h_ln_1_g_0.resize(n_embd);
    h_ln_2_b_0.resize(n_embd);
    h_ln_2_g_0.resize(n_embd);
    h_mlp_c_fc_w_0.resize(n_embd * 4 * n_embd);
    h_mlp_c_fc_b_0.resize(4 * n_embd);
    h_mlp_c_proj_w_0.resize(4 * n_embd * n_embd);
    h_mlp_c_proj_b_0.resize(n_embd);

    // Allocate memory on device
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_w_0, n_embd * 3 * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_b_0, 3 * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_proj_w_0, n_embd * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_proj_b_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_1_b_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_1_g_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_2_b_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_2_g_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_fc_w_0, n_embd * 4 * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_fc_b_0, 4 * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_proj_w_0, 4 * n_embd * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_proj_b_0, n_embd * sizeof(float)));
}

Layer::~Layer() {
    // Free memory on device
    CHECK_CUDA(cudaFree(d_attn_c_attn_w_0));
    CHECK_CUDA(cudaFree(d_attn_c_attn_b_0));
    CHECK_CUDA(cudaFree(d_attn_c_proj_w_0));
    CHECK_CUDA(cudaFree(d_attn_c_proj_b_0));
    CHECK_CUDA(cudaFree(d_ln_1_b_0));
    CHECK_CUDA(cudaFree(d_ln_1_g_0));
    CHECK_CUDA(cudaFree(d_ln_2_b_0));
    CHECK_CUDA(cudaFree(d_ln_2_g_0));
    CHECK_CUDA(cudaFree(d_mlp_c_fc_w_0));
    CHECK_CUDA(cudaFree(d_mlp_c_fc_b_0));
    CHECK_CUDA(cudaFree(d_mlp_c_proj_w_0));
    CHECK_CUDA(cudaFree(d_mlp_c_proj_b_0));
}

__device__ __host__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void multi_head_attention(float* input, float* output, float* w_qkv, float* b_qkv, 
                                     float* w_proj, float* b_proj, int seq_length, int d_model, int n_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_length) return;
    
    // Calculate dimensions
    int d_k = d_model / n_head;
    int qkv_size = 3 * d_model;
    
    // Allocate shared memory dynamically
    extern __shared__ float shared_mem[];
    float* qkv = shared_mem;                                // Size: seq_length * 3 * d_model
    float* out_heads = &qkv[seq_length * qkv_size];         // Size: n_head * seq_length * d_k
    
    __syncthreads();
    
    // Linear projection to get Q, K, V
    for (int i = 0; i < qkv_size; i++) {
        float val = b_qkv[i];
        for (int j = 0; j < d_model; j++) {
            val += input[idx * d_model + j] * w_qkv[i * d_model + j];
        }
        qkv[idx * qkv_size + i] = val;
    }
    
    __syncthreads();
    
    // Scaled dot-product attention per head
    float scores[64]; // Assumes seq_length <= 64, adjust for generality
    float scale = 1.0f / sqrtf(d_k);
    for (int h = 0; h < n_head; h++) {
        float* q_ptr = &qkv[idx * qkv_size + h * d_k];
        float* k_ptr = &qkv[h * d_k];
        float* v_ptr = &qkv[2 * d_model + h * d_k];
        float* out_ptr = &out_heads[h * seq_length * d_k + idx * d_k];
        
        float max_val = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            scores[j] = 0.0f;
            for (int d = 0; d < d_k; d++) {
                scores[j] += q_ptr[d] * k_ptr[j * d_k + d];
            }
            scores[j] *= scale;
            max_val = fmaxf(max_val, scores[j]);
        }
        
        for (int j = 0; j < seq_length; j++) {
            scores[j] = expf(scores[j] - max_val);
            sum += scores[j];
        }
        
        for (int j = 0; j < seq_length; j++) {
            scores[j] /= sum;
        }
        
        for (int d = 0; d < d_k; d++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < seq_length; j++) {
                weighted_sum += scores[j] * v_ptr[j * d_k + d];
            }
            out_ptr[d] = weighted_sum;
        }
    }
    
    __syncthreads();
    
    // Linear projection to get final output
    for (int i = 0; i < d_model; i++) {
        float val = b_proj[i];
        for (int j = 0; j < d_model; j++) {
            val += out_heads[idx * d_model + j] * w_proj[i * d_model + j];
        }
        output[idx * d_model + i] = val;
    }
}

__global__ void layer_normalization(float* input, float* output, float* gamma, float* beta, int seq_length, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_length) return;
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < d_model; i++) {
        mean += input[idx * d_model + i];
    }
    mean /= d_model;
    
    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < d_model; i++) {
        float diff = input[idx * d_model + i] - mean;
        var += diff * diff;
    }
    var /= d_model;
    
    // Normalize and scale
    const float epsilon = 1e-5f;
    float inv_std = 1.0f / sqrtf(var + epsilon);
    
    for (int i = 0; i < d_model; i++) {
        float normalized = (input[idx * d_model + i] - mean) * inv_std;
        output[idx * d_model + i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void mlp(float* input, float* output, float* w_fc, float* b_fc, 
                    float* w_proj, float* b_proj, int seq_length, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_length) return;

    int intermediate_size = 4 * d_model;
    float intermediate[4096]; // Assuming max intermediate_size = 4096, adjust as needed

    // Compute feedforward layer
    for (int i = 0; i < intermediate_size; i++) {
        float val = b_fc[i];
        for (int j = 0; j < d_model; j++) {
            val += input[idx * d_model + j] * w_fc[j * intermediate_size + i];
        }
        intermediate[i] = gelu(val);
    }

    // Compute projection back to hidden size
    for (int i = 0; i < d_model; i++) {
        float val = b_proj[i];
        for (int j = 0; j < intermediate_size; j++) {
            val += intermediate[j] * w_proj[j * d_model + i];
        }
        output[idx * d_model + i] = val;
    }
}

__global__ void add_residual(float* input, float* residual, float* output, int seq_length, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_length) return;
    
    for (int i = 0; i < d_model; i++) {
        output[idx * d_model + i] = input[idx * d_model + i] + residual[idx * d_model + i];
    }
}

void Layer::apply(float* d_hidden_states, float* d_residual, float* d_temp, int seq_length) {
    // Calculate dimensions
    int block_size = 256; // Using a fixed block size that works well for most cases
    int grid_size = (seq_length * n_embd + block_size - 1) / block_size;
    int d_k = n_embd / n_head;
    
    // Calculate shared memory for multi-head attention
    int qkv_size = 3 * n_embd;
    int attn_shared_mem = seq_length * qkv_size +          // qkv
                          3 * n_head * seq_length * d_k +  // q/k/v heads
                          n_head * seq_length * d_k;       // out_heads
    
    // Step 1: Save input for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 2: First layer normalization
    layer_normalization<<<grid_size, block_size>>>(d_hidden_states, d_temp, d_ln_1_g_0, d_ln_1_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 3: Multi-head attention
    multi_head_attention<<<grid_size, block_size, attn_shared_mem * sizeof(float)>>>(
        d_temp, d_hidden_states, 
        d_attn_c_attn_w_0, d_attn_c_attn_b_0, 
        d_attn_c_proj_w_0, d_attn_c_proj_b_0, 
        seq_length, n_embd, n_head);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 4: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 6: Second layer normalization
    layer_normalization<<<grid_size, block_size>>>(d_hidden_states, d_temp, d_ln_2_g_0, d_ln_2_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 7: MLP (feedforward network)
    mlp<<<grid_size, block_size>>>(
        d_temp, d_hidden_states,
        d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
        seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 8: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Final synchronization (only if needed)
    // CHECK_CUDA(cudaDeviceSynchronize());
}

void Layer::load_from_hdf5(hid_t file_id, const std::string& layer_path) {
    read_dataset(file_id, layer_path + "/attn/c_attn/w_0", h_attn_c_attn_w_0);
    read_dataset(file_id, layer_path + "/attn/c_attn/b_0", h_attn_c_attn_b_0);
    read_dataset(file_id, layer_path + "/attn/c_proj/w_0", h_attn_c_proj_w_0);
    read_dataset(file_id, layer_path + "/attn/c_proj/b_0", h_attn_c_proj_b_0);
    read_dataset(file_id, layer_path + "/ln_1/b_0",        h_ln_1_b_0);
    read_dataset(file_id, layer_path + "/ln_1/g_0",        h_ln_1_g_0);
    read_dataset(file_id, layer_path + "/ln_2/b_0",        h_ln_2_b_0);
    read_dataset(file_id, layer_path + "/ln_2/g_0",        h_ln_2_g_0);
    read_dataset(file_id, layer_path + "/mlp/c_fc/w_0",    h_mlp_c_fc_w_0);
    read_dataset(file_id, layer_path + "/mlp/c_fc/b_0",    h_mlp_c_fc_b_0);
    read_dataset(file_id, layer_path + "/mlp/c_proj/w_0",  h_mlp_c_proj_w_0);
    read_dataset(file_id, layer_path + "/mlp/c_proj/b_0",  h_mlp_c_proj_b_0);
}

void Layer::copy_host_to_device() {
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_w_0, h_attn_c_attn_w_0.data(), h_attn_c_attn_w_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_b_0, h_attn_c_attn_b_0.data(), h_attn_c_attn_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_proj_w_0, h_attn_c_proj_w_0.data(), h_attn_c_proj_w_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_proj_b_0, h_attn_c_proj_b_0.data(), h_attn_c_proj_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_1_b_0, h_ln_1_b_0.data(), h_ln_1_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_1_g_0, h_ln_1_g_0.data(), h_ln_1_g_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_2_b_0, h_ln_2_b_0.data(), h_ln_2_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_2_g_0, h_ln_2_g_0.data(), h_ln_2_g_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_fc_w_0, h_mlp_c_fc_w_0.data(), h_mlp_c_fc_w_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_fc_b_0, h_mlp_c_fc_b_0.data(), h_mlp_c_fc_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_proj_w_0, h_mlp_c_proj_w_0.data(), h_mlp_c_proj_w_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_proj_b_0, h_mlp_c_proj_b_0.data(), h_mlp_c_proj_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void Layer::copy_device_to_host() {
    CHECK_CUDA(cudaMemcpy(h_attn_c_attn_w_0.data(), d_attn_c_attn_w_0, h_attn_c_attn_w_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_attn_b_0.data(), d_attn_c_attn_b_0, h_attn_c_attn_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_proj_w_0.data(), d_attn_c_proj_w_0, h_attn_c_proj_w_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_proj_b_0.data(), d_attn_c_proj_b_0, h_attn_c_proj_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_1_b_0.data(), d_ln_1_b_0, h_ln_1_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_1_g_0.data(), d_ln_1_g_0, h_ln_1_g_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_2_b_0.data(), d_ln_2_b_0, h_ln_2_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_2_g_0.data(), d_ln_2_g_0, h_ln_2_g_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_fc_w_0.data(), d_mlp_c_fc_w_0, h_mlp_c_fc_w_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_fc_b_0.data(), d_mlp_c_fc_b_0, h_mlp_c_fc_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_proj_w_0.data(), d_mlp_c_proj_w_0, h_mlp_c_proj_w_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_proj_b_0.data(), d_mlp_c_proj_b_0, h_mlp_c_proj_b_0.size() * sizeof(float), cudaMemcpyDeviceToHost));
}