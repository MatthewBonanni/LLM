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
    
    // Allocate shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* qkv = shared_mem;                                // Size: seq_length * 3 * d_model
    float* q_heads = &qkv[seq_length * qkv_size];           // Size: n_head * seq_length * d_k
    float* k_heads = &q_heads[n_head * seq_length * d_k];   // Size: n_head * seq_length * d_k
    float* v_heads = &k_heads[n_head * seq_length * d_k];   // Size: n_head * seq_length * d_k
    float* out_heads = &v_heads[n_head * seq_length * d_k]; // Size: n_head * seq_length * d_k
    
    // Linear projection to get Q, K, V
    for (int i = 0; i < qkv_size; i++) {
        float val = b_qkv[i];
        for (int j = 0; j < d_model; j++) {
            val += input[idx * d_model + j] * w_qkv[j * qkv_size + i];
        }
        qkv[idx * qkv_size + i] = val;
    }
    
    // Split heads
    for (int h = 0; h < n_head; h++) {
        for (int d = 0; d < d_k; d++) {
            q_heads[h * seq_length * d_k + idx * d_k + d] = qkv[idx * qkv_size + h * d_k + d];
            k_heads[h * seq_length * d_k + idx * d_k + d] = qkv[idx * qkv_size + d_model + h * d_k + d];
            v_heads[h * seq_length * d_k + idx * d_k + d] = qkv[idx * qkv_size + 2 * d_model + h * d_k + d];
        }
    }
    
    __syncthreads();
    
    // Apply scaled dot-product attention for each head
    for (int h = 0; h < n_head; h++) {
        float* q_head = &q_heads[h * seq_length * d_k];
        float* k_head = &k_heads[h * seq_length * d_k];
        float* v_head = &v_heads[h * seq_length * d_k];
        float* out_head = &out_heads[h * seq_length * d_k];
        
        // Compute attention scores and output
        float* q_ptr = &q_head[idx * d_k];
        float* out_ptr = &out_head[idx * d_k];
        
        // Calculate attention scores
        float scores[64]; // Assuming max seq_length = 64, adjust as needed
        float scale = 1.0f / sqrtf(d_k);
        
        for (int j = 0; j < seq_length; j++) {
            float score = 0.0f;
            float* k_ptr = &k_head[j * d_k];
            for (int d = 0; d < d_k; d++) {
                score += q_ptr[d] * k_ptr[d];
            }
            scores[j] = score * scale;
        }
        
        // Apply softmax
        float max_val = -INFINITY;
        for (int j = 0; j < seq_length; j++) {
            max_val = fmaxf(max_val, scores[j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            scores[j] = expf(scores[j] - max_val);
            sum += scores[j];
        }
        
        for (int j = 0; j < seq_length; j++) {
            scores[j] /= sum;
        }
        
        // Calculate weighted values
        for (int d = 0; d < d_k; d++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < seq_length; j++) {
                weighted_sum += scores[j] * v_head[j * d_k + d];
            }
            out_ptr[d] = weighted_sum;
        }
    }
    
    __syncthreads();
    
    // Concatenate heads
    float concat_heads[1024]; // Assuming max d_model = 1024, adjust as needed
    for (int h = 0; h < n_head; h++) {
        for (int d = 0; d < d_k; d++) {
            concat_heads[h * d_k + d] = out_heads[h * seq_length * d_k + idx * d_k + d];
        }
    }
    
    // Linear projection to get output
    for (int i = 0; i < d_model; i++) {
        float val = b_proj[i];
        for (int j = 0; j < d_model; j++) {
            val += concat_heads[j] * w_proj[j * d_model + i];
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

void Layer::apply(float* d_hidden_states, float* d_residual, int seq_length) {
    // Calculate dimensions and shared memory requirements
    int shared_mem_size = 0;
    
    // For multi-head attention:
    // - qkv: seq_length * 3 * n_embd
    // - q/k/v heads: 3 * n_head * seq_length * (n_embd / n_head)
    // - out_heads: n_head * seq_length * (n_embd / n_head)
    int qkv_size = 3 * n_embd;
    int d_k = n_embd / n_head;
    int attn_shared_mem = seq_length * qkv_size +          // qkv
                          3 * n_head * seq_length * d_k +  // q/k/v heads
                          n_head * seq_length * d_k;       // out_heads
    shared_mem_size = max(shared_mem_size, attn_shared_mem);
    
    // Prepare temp buffers
    float* d_temp = nullptr;
    CHECK_CUDA(cudaMalloc(&d_temp, seq_length * n_embd * sizeof(float)));
    
    // Step 1: Save input for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 2: First layer normalization
    layer_normalization<<<1, seq_length>>>(d_hidden_states, d_temp, d_ln_1_g_0, d_ln_1_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 3: Multi-head attention
    int block_size = 32;
    int grid_size = (seq_length + block_size - 1) / block_size;
    multi_head_attention<<<grid_size, block_size, shared_mem_size * sizeof(float)>>>(
        d_temp, d_hidden_states, 
        d_attn_c_attn_w_0, d_attn_c_attn_b_0, 
        d_attn_c_proj_w_0, d_attn_c_proj_b_0, 
        seq_length, n_embd, n_head);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 4: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 6: Second layer normalization
    layer_normalization<<<grid_size, block_size>>>(d_hidden_states, d_temp, d_ln_2_g_0, d_ln_2_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 7: MLP (feedforward network)
    mlp<<<grid_size, block_size>>>(
        d_temp, d_hidden_states,
        d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
        seq_length, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Step 8: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Free temporary buffer
    CHECK_CUDA(cudaFree(d_temp));
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