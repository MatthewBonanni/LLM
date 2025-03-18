#include "layer.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

#include "utils.cuh"
#include "io.cuh"

#define INTERMEDIATE_SIZE_MAX 4096
#define SEQ_LENGTH_MAX 2048

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

__global__ void multi_head_attention(float* input, float* output,
                                     float* w_qkv, float* b_qkv, 
                                     float* w_proj, float* b_proj,
                                     int seq_length, int n_embd, int n_head,
                                     float* qkv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    // Calculate dimensions
    int d_k = n_embd / n_head; // Dimension of each head
    int qkv_size = 3 * n_embd; // Size of Q, K, V for each token

    // Scores register
    float scores[SEQ_LENGTH_MAX];

    // Linear projection to get Q, K, V for this token
    for (int i = 0; i < qkv_size; i++) {
        float val = b_qkv[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[i_token * n_embd + j] * w_qkv[j * qkv_size + i];
        }
        qkv[i_token * qkv_size + i] = val;
    }

    // Ensure all threads have completed their QKV calculations
    // Prevent race condition in the following loop
    __syncthreads();

    float scale = 1.0f / sqrtf(d_k);

    // Process each attention head
    for (int i_head = 0; i_head < n_head; i_head++) {
        // Calculate attention scores between current token and all other tokens
        float max_val = -INFINITY;
        for (int j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: only attend to positions j_token <= i_token
            if (j_token <= i_token) {
                // Compute dot product
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    // Q values for token i_token, head i_head
                    // K values for token j_token, head i_head
                    dot += qkv[i_token * qkv_size              + i_head * d_k + d] *
                           qkv[j_token * qkv_size + 1 * n_embd + i_head * d_k + d];
                }
                scores[j_token] = dot * scale;
                max_val = fmaxf(max_val, scores[j_token]);
            }
        }

        // Softmax calculation for attention weights
        float sum = 0.0f;
        for (int j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: masked tokens have zero weight
            if (j_token <= i_token) {
                scores[j_token] = expf(scores[j_token] - max_val);
                sum += scores[j_token];
            } else {
                scores[j_token] = 0.0f;
            }
        }

        for (int j_token = 0; j_token < seq_length; j_token++) {
            scores[j_token] /= sum;
        }

        // Calculate weighted sum of values
        for (int d = 0; d < d_k; d++) {
            float weighted_sum = 0.0f;
            for (int j_token = 0; j_token < seq_length; j_token++) {
                // Get V values for token j_token, head i_head
                weighted_sum += scores[j_token] *
                                qkv[j_token * qkv_size + 2 * n_embd + i_head * d_k + d];
            }
            // Use input as a temporary buffer to store head outputs
            input[i_head * seq_length * d_k + i_token * d_k + d] = weighted_sum;
        }
    }

    // Linear projection to get final output
    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int i_head = 0; i_head < n_head; i_head++) {
            for (int d = 0; d < d_k; d++) {
                int j = i_head * d_k + d;  // Concatenated head outputs
                val += input[i_head * seq_length * d_k + i_token * d_k + d] *
                       w_proj[i * n_embd + j];
            }
        }
        output[i_token * n_embd + i] = val;
    }
}

__global__ void layer_normalization(float* input, float* output, float* gamma, float* beta, int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        mean += input[i_token * n_embd + i];
    }
    mean /= n_embd;
    
    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        float diff = input[i_token * n_embd + i] - mean;
        var += diff * diff;
    }
    var /= n_embd;
    
    // Normalize and scale
    const float epsilon = 1e-5f;
    float inv_std = 1.0f / sqrtf(var + epsilon);
    
    for (int i = 0; i < n_embd; i++) {
        float normalized = (input[i_token * n_embd + i] - mean) * inv_std;
        output[i_token * n_embd + i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void mlp(float* input, float* output, float* w_fc, float* b_fc, 
                    float* w_proj, float* b_proj, int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    int intermediate_size = 4 * n_embd;
    float intermediate[INTERMEDIATE_SIZE_MAX];

    // Compute feedforward layer
    for (int i = 0; i < intermediate_size; i++) {
        float val = b_fc[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[i_token * n_embd + j] * w_fc[j * intermediate_size + i];
        }
        intermediate[i] = gelu(val);
    }

    // Compute projection back to hidden size
    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int j = 0; j < intermediate_size; j++) {
            val += intermediate[j] * w_proj[j * n_embd + i];
        }
        output[i_token * n_embd + i] = val;
    }
}

__global__ void add_residual(float* input, float* residual, float* output, int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }
    
    for (int i = 0; i < n_embd; i++) {
        output[i_token * n_embd + i] = input[i_token * n_embd + i] + residual[i_token * n_embd + i];
    }
}

void Layer::apply(float* d_hidden_states, float* d_residual, float* d_temp, int seq_length) {
    // Calculate dimensions
    int block_size = 256; // Using a fixed block size that works well for most cases
    int grid_size = (seq_length * n_embd + block_size - 1) / block_size;

    // Allocate temporary buffers
    float* d_qkv = nullptr;
    CHECK_CUDA(cudaMalloc(&d_qkv, seq_length * 3 * n_embd * sizeof(float)));
    
    // Step 1: Save input for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 2: First layer normalization
    layer_normalization<<<grid_size, block_size>>>(d_hidden_states, d_temp, d_ln_1_g_0, d_ln_1_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3: Multi-head attention
    multi_head_attention<<<grid_size, block_size>>>(
        d_temp, d_hidden_states, 
        d_attn_c_attn_w_0, d_attn_c_attn_b_0, 
        d_attn_c_proj_w_0, d_attn_c_proj_b_0, 
        seq_length, n_embd, n_head,
        d_qkv);
    CHECK_CUDA(cudaGetLastError());
    
    // DEBUG
    int n_print = 5;
    // Transfer QKV to host and print
    std::vector<float> h_qkv(seq_length * 3 * n_embd);
    CHECK_CUDA(cudaMemcpy(h_qkv.data(), d_qkv, seq_length * 3 * n_embd * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Q:" << std::endl;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < n_print; j++) {
            std::cout << h_qkv[i * 3 * n_embd + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "K:" << std::endl;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < n_print; j++) {
            std::cout << h_qkv[i * 3 * n_embd + n_embd + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "V:" << std::endl;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < n_print; j++) {
            std::cout << h_qkv[i * 3 * n_embd + 2 * n_embd + j] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> h_hidden_states(seq_length * n_embd);

    // Transfer temp to host and print them
    CHECK_CUDA(cudaMemcpy(h_hidden_states.data(), d_temp, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Temp:" << std::endl;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < n_print; j++) {
            std::cout << h_hidden_states[i * n_embd + j] << " ";
        }
        std::cout << std::endl;
    }

    // Transfer hidden states to host and print them
    CHECK_CUDA(cudaMemcpy(h_hidden_states.data(), d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Hidden States:" << std::endl;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < n_print; j++) {
            std::cout << h_hidden_states[i * n_embd + j] << " ";
        }
        std::cout << std::endl;
    }
    exit(0);
    
    // Step 4: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 6: Second layer normalization
    layer_normalization<<<grid_size, block_size>>>(d_hidden_states, d_temp, d_ln_2_g_0, d_ln_2_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 7: MLP (feedforward network)
    mlp<<<grid_size, block_size>>>(d_temp, d_hidden_states,
                                   d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
                                   d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
                                   seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
    
    // Step 8: Add residual connection
    add_residual<<<grid_size, block_size>>>(d_hidden_states, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Free temporary buffers
    CHECK_CUDA(cudaFree(d_qkv));
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