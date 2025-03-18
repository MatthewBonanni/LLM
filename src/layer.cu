#include "layer.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

#include "utils.cuh"
#include "io.cuh"
#include "kernels.cuh"

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
    layer_normalization_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_ln_1_g_0, d_ln_1_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3: Multi-head attention
    // Step 3.1: QKV projection
    qkv_projection_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_qkv, 
        d_attn_c_attn_w_0, d_attn_c_attn_b_0, 
        seq_length, n_embd, 3 * n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3.2: Multi-head attention
    multi_head_attention_kernel<<<grid_size, block_size>>>(
        d_qkv, d_hidden_states, 
        seq_length, n_embd, n_head);
    CHECK_CUDA(cudaGetLastError());

    // Step 3.3: Final projection
    final_projection_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_temp,
        d_attn_c_proj_w_0, d_attn_c_proj_b_0,
        seq_length, n_embd);

    // Step 4: Add residual connection
    add_residual_kernel<<<grid_size, block_size>>>(
        d_temp, d_residual, d_hidden_states, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(d_residual, d_hidden_states, seq_length * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));

    // Step 6: Second layer normalization
    layer_normalization_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_ln_2_g_0, d_ln_2_b_0, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 7: MLP (feedforward network)
    mlp_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_temp,
        d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
        seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 8: Add residual connection
    add_residual_kernel<<<grid_size, block_size>>>(
        d_temp, d_residual, d_hidden_states, seq_length, n_embd);
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