#include "layer.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <vector>

#include "utils.cuh"
#include "io.cuh"
#include "kernels.cuh"

Layer::Layer(uint32_t n_ctx, uint32_t n_embd, uint32_t n_head) : 
        n_ctx(n_ctx),
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
        d_mlp_c_proj_b_0(nullptr),
        d_kv_cache(nullptr),
        kv_cache_size(0) {
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
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_w_0, n_embd * 3 * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_b_0, 3 * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_proj_w_0, n_embd * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_proj_b_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_ln_1_b_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_ln_1_g_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_ln_2_b_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_ln_2_g_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_fc_w_0, n_embd * 4 * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_fc_b_0, 4 * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_proj_w_0, 4 * n_embd * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_mlp_c_proj_b_0, n_embd * sizeof(fp_t)));
}

Layer::~Layer() {
    // Free memory on device
    std::vector<void*> buffers = {
        d_attn_c_attn_w_0,
        d_attn_c_attn_b_0,
        d_attn_c_proj_w_0,
        d_attn_c_proj_b_0,
        d_ln_1_b_0,
        d_ln_1_g_0,
        d_ln_2_b_0,
        d_ln_2_g_0,
        d_mlp_c_fc_w_0,
        d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0,
        d_mlp_c_proj_b_0,
        d_kv_cache
    };
    clean_up_memory(buffers);
}

void Layer::allocate_kv_cache(uint32_t batch_size) {
    CHECK_CUDA(cudaMalloc(&d_kv_cache, (uint64_t)batch_size * n_ctx * 2 * n_embd * sizeof(half)));
}

void Layer::apply(
        fp_t* d_hidden_states,
        fp_t* d_residual,
        fp_t* d_temp,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset) {
    // Dimensions
    dim3 grid_size;
    dim3 block_size;

    // Allocate temporary Q buffer
    fp_t* d_q = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, (uint64_t)batch_size * seq_length * n_embd * sizeof(fp_t)));
    
    // Step 1: Save input for residual connection
    CHECK_CUDA(cudaMemcpy(
        d_residual,
        d_hidden_states,
        (uint64_t)batch_size * seq_length * n_embd * sizeof(fp_t),
        cudaMemcpyDeviceToDevice));
    
    // Step 2: First layer normalization
    // Each block handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = 256;
    block_size.y = 1;
    block_size.z = 1;
    grid_size.x = batch_size;
    grid_size.y = seq_length;
    grid_size.z = 1;
    layer_normalization_kernel<256, 8><<<grid_size, block_size>>>(
        d_hidden_states, d_ln_1_g_0, d_ln_1_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3: Multi-head attention
    // Step 3.1: QKV projection
    block_size.x = 32;
    block_size.y = 16;
    block_size.z = 1;
    grid_size.x = (n_embd     + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = batch_size;
    size_t shared_mem_size = (WMMA_M * n_embd + 3 * WMMA_M * WMMA_K) * sizeof(half);
    qkv_projection_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_hidden_states, d_q, d_kv_cache,
        d_attn_c_attn_w_0, d_attn_c_attn_b_0, 
        batch_size, seq_length, seq_offset, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3.2: Multi-head attention
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = std::min(batch_size, (uint32_t)32);
    block_size.y = std::min(seq_length, (uint32_t)32);
    block_size.z = 1;
    grid_size.x = (batch_size + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    multi_head_attention_kernel<<<grid_size, block_size>>>(
        d_q, d_kv_cache, d_hidden_states, 
        batch_size, seq_length, seq_offset, n_head, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 3.3: Final projection
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = std::min(batch_size, (uint32_t)32);
    block_size.y = std::min(seq_length, (uint32_t)32);
    block_size.z = 1;
    grid_size.x = (batch_size + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    final_projection_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_temp,
        d_attn_c_proj_w_0, d_attn_c_proj_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 4: Add residual connection
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = std::min(batch_size, (uint32_t)32);
    block_size.y = std::min(seq_length, (uint32_t)32);
    block_size.z = 1;
    grid_size.x = (batch_size + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    add_residual_kernel<<<grid_size, block_size>>>(
        d_temp, d_residual, d_hidden_states,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(
        d_residual,
        d_hidden_states,
        (uint64_t)seq_length * n_embd * sizeof(fp_t),
        cudaMemcpyDeviceToDevice));

    // Step 6: Second layer normalization
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = 256;
    block_size.y = 1;
    block_size.z = 1;
    grid_size.x = batch_size;
    grid_size.y = seq_length;
    grid_size.z = 1;
    layer_normalization_kernel<256, 8><<<grid_size, block_size>>>(
        d_hidden_states, d_ln_1_g_0, d_ln_1_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 7: MLP (feedforward network)
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = std::min(batch_size, (uint32_t)32);
    block_size.y = std::min(seq_length, (uint32_t)32);
    block_size.z = 1;
    grid_size.x = (batch_size + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    mlp_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_temp,
        d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Step 8: Add residual connection
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    block_size.x = std::min(batch_size, (uint32_t)32);
    block_size.y = std::min(seq_length, (uint32_t)32);
    block_size.z = 1;
    grid_size.x = (batch_size + block_size.x - 1) / block_size.x;
    grid_size.y = (seq_length + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    add_residual_kernel<<<grid_size, block_size>>>(
        d_temp, d_residual, d_hidden_states,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // Free temporary buffers
    std::vector<void*> buffers = {d_q};
    clean_up_memory(buffers);
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
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_w_0, h_attn_c_attn_w_0.data(), h_attn_c_attn_w_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_b_0, h_attn_c_attn_b_0.data(), h_attn_c_attn_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_proj_w_0, h_attn_c_proj_w_0.data(), h_attn_c_proj_w_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_proj_b_0, h_attn_c_proj_b_0.data(), h_attn_c_proj_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_1_b_0, h_ln_1_b_0.data(), h_ln_1_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_1_g_0, h_ln_1_g_0.data(), h_ln_1_g_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_2_b_0, h_ln_2_b_0.data(), h_ln_2_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_2_g_0, h_ln_2_g_0.data(), h_ln_2_g_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_fc_w_0, h_mlp_c_fc_w_0.data(), h_mlp_c_fc_w_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_fc_b_0, h_mlp_c_fc_b_0.data(), h_mlp_c_fc_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_proj_w_0, h_mlp_c_proj_w_0.data(), h_mlp_c_proj_w_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mlp_c_proj_b_0, h_mlp_c_proj_b_0.data(), h_mlp_c_proj_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
}

void Layer::copy_device_to_host() {
    CHECK_CUDA(cudaMemcpy(h_attn_c_attn_w_0.data(), d_attn_c_attn_w_0, h_attn_c_attn_w_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_attn_b_0.data(), d_attn_c_attn_b_0, h_attn_c_attn_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_proj_w_0.data(), d_attn_c_proj_w_0, h_attn_c_proj_w_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_attn_c_proj_b_0.data(), d_attn_c_proj_b_0, h_attn_c_proj_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_1_b_0.data(), d_ln_1_b_0, h_ln_1_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_1_g_0.data(), d_ln_1_g_0, h_ln_1_g_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_2_b_0.data(), d_ln_2_b_0, h_ln_2_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ln_2_g_0.data(), d_ln_2_g_0, h_ln_2_g_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_fc_w_0.data(), d_mlp_c_fc_w_0, h_mlp_c_fc_w_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_fc_b_0.data(), d_mlp_c_fc_b_0, h_mlp_c_fc_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_proj_w_0.data(), d_mlp_c_proj_w_0, h_mlp_c_proj_w_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mlp_c_proj_b_0.data(), d_mlp_c_proj_b_0, h_mlp_c_proj_b_0.size() * sizeof(fp_t), cudaMemcpyDeviceToHost));
}