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
        d_attn_c_attn_w_Q_0(nullptr),
        d_attn_c_attn_w_K_0(nullptr),
        d_attn_c_attn_w_V_0(nullptr),
        d_attn_c_attn_b_Q_0(nullptr),
        d_attn_c_attn_b_K_0(nullptr),
        d_attn_c_attn_b_V_0(nullptr),
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
        d_k_cache(nullptr),
        d_v_cache(nullptr),
        kv_cache_size(0) {
    // Allocate memory on device
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_w_Q_0, n_embd * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_w_K_0, n_embd * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_w_V_0, n_embd * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_b_Q_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_b_K_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_attn_c_attn_b_V_0, n_embd * sizeof(fp_t)));
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
        d_attn_c_attn_w_Q_0,
        d_attn_c_attn_w_K_0,
        d_attn_c_attn_w_V_0,
        d_attn_c_attn_b_Q_0,
        d_attn_c_attn_b_K_0,
        d_attn_c_attn_b_V_0,
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
        d_k_cache,
        d_v_cache
    };
    clean_up_memory(buffers);
}

void Layer::allocate_kv_cache(uint32_t batch_size) {
    CHECK_CUDA(cudaMalloc(&d_k_cache, (uint64_t)batch_size * n_ctx * n_embd * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_v_cache, (uint64_t)batch_size * n_ctx * n_embd * sizeof(half)));
}

void Layer::launch_layer_normalization(
        fp_t* d_input,
        uint32_t batch_size,
        uint32_t seq_length) {
    // Each block handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 grid_size(batch_size, seq_length, 1);
    dim3 block_size(256, 1, 1);
    layer_normalization_kernel<256, 8><<<grid_size, block_size>>>(
        d_input, d_ln_1_g_0, d_ln_1_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void Layer::launch_qkv_projection(
        const fp_t* d_hidden_states,
        fp_t* d_q,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset) {
    dim3 block_size_q(32, 32, 1);
    dim3 grid_size_q((n_embd     + block_size_q.x - 1) / block_size_q.x,
                     (batch_size + block_size_q.y - 1) / block_size_q.y,
                     1);
    q_projection_kernel<<<grid_size_q, block_size_q>>>(
        d_hidden_states, d_q,
        d_attn_c_attn_w_Q_0, d_attn_c_attn_b_Q_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // // DEBUG
    // std::vector<fp_t> h_q(batch_size * n_embd);
    // CHECK_CUDA(cudaMemcpy(h_q.data(), d_q, batch_size * n_embd * sizeof(fp_t), cudaMemcpyDeviceToHost));
    // std::cout << "Q: " << std::endl;
    // for (uint32_t i = 0; i < batch_size; i++) {
    //     for (uint32_t j = 0; j < 5; j++) {
    //         std::cout << h_q[i * n_embd + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "------------------" << std::endl;


    // constexpr uint32_t BLOCK_M_Q = 32;
    // constexpr uint32_t BLOCK_N_Q = 32;
    // constexpr uint32_t BLOCK_K_Q = 32;
    // constexpr uint32_t WARPS_PER_BLOCK_Q = (BLOCK_M_Q/WMMA_M * BLOCK_N_Q/WMMA_N);
    // dim3 grid_size_q(((n_embd     + BLOCK_N_Q - 1) / BLOCK_N_Q),
    //                  ((batch_size + BLOCK_M_Q - 1) / BLOCK_M_Q),
    //                  1);
    // dim3 block_size_q(WARP_SIZE, WARPS_PER_BLOCK_Q, 1);
    // uint32_t shared_mem_size_q = (BLOCK_M_Q * BLOCK_K_Q + BLOCK_K_Q * BLOCK_N_Q) * sizeof(half);
    // q_projection_kernel<BLOCK_M_Q, BLOCK_N_Q, BLOCK_K_Q><<<grid_size_q, block_size_q, shared_mem_size_q>>>(
    //     d_hidden_states, d_q,
    //     d_attn_c_attn_w_Q_0, d_attn_c_attn_b_Q_0,
    //     batch_size, seq_length, n_embd);
    // CHECK_CUDA(cudaGetLastError());

    dim3 block_size_kv(32, 32, 1);
    dim3 grid_size_kv((n_embd     + block_size_kv.x - 1) / block_size_kv.x,
                      (seq_length + block_size_kv.y - 1) / block_size_kv.y,
                      batch_size);
    kv_projection_kernel<<<grid_size_kv, block_size_kv>>>(
        d_hidden_states, d_k_cache,
        d_attn_c_attn_w_K_0, d_attn_c_attn_b_K_0,
        batch_size, seq_length, seq_offset, n_embd);
    CHECK_CUDA(cudaGetLastError());
    kv_projection_kernel<<<grid_size_kv, block_size_kv>>>(
        d_hidden_states, d_v_cache,
        d_attn_c_attn_w_V_0, d_attn_c_attn_b_V_0,
        batch_size, seq_length, seq_offset, n_embd);
    CHECK_CUDA(cudaGetLastError());

    // // DEBUG
    // std::vector<half> h_k(batch_size * seq_length * n_embd);
    // CHECK_CUDA(cudaMemcpy(h_k.data(), d_k_cache, batch_size * seq_length * n_embd * sizeof(half), cudaMemcpyDeviceToHost));
    // std::cout << "seq_length: " << seq_length << std::endl;
    // std::cout << "seq_offset: " << seq_offset << std::endl;
    // std::cout << "K cache: " << std::endl;
    // for (uint32_t i = 0; i < batch_size; i++) {
    //     for (uint32_t j = 0; j < seq_length; j++) {
    //         for (uint32_t k = 0; k < 5; k++) {
    //             std::cout << __half2float(h_k[i * seq_length * n_embd + (j + seq_offset) * n_embd + k]) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "------------------" << std::endl;
    // }
    // exit(0);


    // constexpr uint32_t BLOCK_M_KV = 32;
    // constexpr uint32_t BLOCK_N_KV = 32;
    // constexpr uint32_t BLOCK_K_KV = 32;
    // constexpr uint32_t WARPS_PER_BLOCK_KV = (BLOCK_M_KV/WMMA_M * BLOCK_N_KV/WMMA_N);
    // dim3 grid_size_kv(((n_embd     + BLOCK_N_KV - 1) / BLOCK_N_KV),
    //                   ((seq_length + BLOCK_M_KV - 1) / BLOCK_M_KV),
    //                   batch_size);
    // dim3 block_size_kv(WARP_SIZE, WARPS_PER_BLOCK_KV, 1);
    // uint32_t shared_mem_size_kv = std::max(
    //     (BLOCK_M_KV * BLOCK_K_KV + BLOCK_K_KV * BLOCK_N_KV) * sizeof(half),
    //     (BLOCK_M_KV * BLOCK_N_KV) * sizeof(fp_t));
    // kv_projection_kernel<BLOCK_M_KV, BLOCK_N_KV, BLOCK_K_KV><<<grid_size_kv, block_size_kv, shared_mem_size_kv>>>(
    //     d_hidden_states, d_k_cache,
    //     d_attn_c_attn_w_K_0, d_attn_c_attn_b_K_0,
    //     batch_size, seq_length, seq_offset, n_embd);
    // CHECK_CUDA(cudaGetLastError());
    // kv_projection_kernel<BLOCK_M_KV, BLOCK_N_KV, BLOCK_K_KV><<<grid_size_kv, block_size_kv, shared_mem_size_kv>>>(
    //     d_hidden_states, d_v_cache,
    //     d_attn_c_attn_w_V_0, d_attn_c_attn_b_V_0,
    //     batch_size, seq_length, seq_offset, n_embd);
    // CHECK_CUDA(cudaGetLastError());
}

void Layer::launch_multi_head_attention(
        const fp_t* d_q,
        fp_t* d_output,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    multi_head_attention_kernel<<<grid_size, block_size>>>(
        d_q, d_k_cache, d_v_cache, d_output,
        batch_size, seq_length, seq_offset, n_head, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void Layer::launch_final_projection(
        const fp_t* d_input,
        fp_t* d_output,
        uint32_t batch_size,
        uint32_t seq_length) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    final_projection_kernel<<<grid_size, block_size>>>(
        d_input, d_output,
        d_attn_c_proj_w_0, d_attn_c_proj_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void Layer::launch_add_residual(
        const fp_t* d_input,
        const fp_t* d_residual,
        fp_t* d_output,
        uint32_t batch_size,
        uint32_t seq_length) {
    // Each block handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(128, 1, 1);
    dim3 grid_size(batch_size, seq_length, 1);
    add_residual_kernel<<<grid_size, block_size>>>(
        d_input, d_residual, d_output,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void Layer::launch_mlp(
        const fp_t* d_input,
        fp_t* d_output,
        uint32_t batch_size,
        uint32_t seq_length) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    mlp_kernel<<<grid_size, block_size>>>(
        d_input, d_output,
        d_mlp_c_fc_w_0, d_mlp_c_fc_b_0,
        d_mlp_c_proj_w_0, d_mlp_c_proj_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void Layer::apply(
        fp_t* d_hidden_states,
        fp_t* d_residual,
        fp_t* d_temp,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t seq_offset) {
    // Allocate temporary Q buffer
    fp_t* d_q = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, (uint64_t)batch_size * n_embd * sizeof(fp_t)));
    
    // Step 1: Save input for residual connection
    CHECK_CUDA(cudaMemcpy(
        d_residual,
        d_hidden_states,
        (uint64_t)batch_size * seq_length * n_embd * sizeof(fp_t),
        cudaMemcpyDeviceToDevice));
    
    // Step 2: First layer normalization
    launch_layer_normalization(d_hidden_states, batch_size, seq_length);

    // Step 3: Multi-head attention
    // Step 3.1: QKV projection
    launch_qkv_projection(d_hidden_states, d_q, batch_size, seq_length, seq_offset);

    // Step 3.2: Multi-head attention
    launch_multi_head_attention(d_q, d_temp, batch_size, seq_length, seq_offset);

    // Step 3.3: Final projection
    launch_final_projection(d_temp, d_hidden_states, batch_size, seq_length);

    // Step 4: Add residual connection
    launch_add_residual(d_temp, d_residual, d_hidden_states, batch_size, seq_length);

    // Step 5: Save output for residual connection
    CHECK_CUDA(cudaMemcpy(
        d_residual,
        d_hidden_states,
        (uint64_t)seq_length * n_embd * sizeof(fp_t),
        cudaMemcpyDeviceToDevice));

    // Step 6: Second layer normalization
    launch_layer_normalization(d_hidden_states, batch_size, seq_length);

    // Step 7: MLP (feedforward network)
    launch_mlp(d_hidden_states, d_temp, batch_size, seq_length);

    // Step 8: Add residual connection
    launch_add_residual(d_temp, d_residual, d_hidden_states, batch_size, seq_length);

    // Free temporary buffer
    std::vector<void*> buffers = {d_q};
    clean_up_memory(buffers);
}

void Layer::load_from_hdf5(hid_t file_id, const std::string& layer_path) {
    // Host buffers
    std::vector<fp_t> h_attn_c_attn_w_0(n_embd * 3 * n_embd);
    std::vector<fp_t> h_attn_c_attn_b_0(3 * n_embd);
    std::vector<fp_t> h_attn_c_proj_w_0(n_embd * n_embd);
    std::vector<fp_t> h_attn_c_proj_b_0(n_embd);
    std::vector<fp_t> h_ln_1_b_0(n_embd);
    std::vector<fp_t> h_ln_1_g_0(n_embd);
    std::vector<fp_t> h_ln_2_b_0(n_embd);
    std::vector<fp_t> h_ln_2_g_0(n_embd);
    std::vector<fp_t> h_mlp_c_fc_w_0(n_embd * 4 * n_embd);
    std::vector<fp_t> h_mlp_c_fc_b_0(4 * n_embd);
    std::vector<fp_t> h_mlp_c_proj_w_0(4 * n_embd * n_embd);
    std::vector<fp_t> h_mlp_c_proj_b_0(n_embd);

    // Read datasets from HDF5 file
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

    // Separate attention weights and biases into Q, K, and V
    std::vector<fp_t> h_attn_c_attn_w_Q_0(n_embd * n_embd);
    std::vector<fp_t> h_attn_c_attn_w_K_0(n_embd * n_embd);
    std::vector<fp_t> h_attn_c_attn_w_V_0(n_embd * n_embd);
    std::vector<fp_t> h_attn_c_attn_b_Q_0(n_embd);
    std::vector<fp_t> h_attn_c_attn_b_K_0(n_embd);
    std::vector<fp_t> h_attn_c_attn_b_V_0(n_embd);
    for (uint32_t i = 0; i < n_embd; i++) {
        for (uint32_t j = 0; j < n_embd; j++) {
            // Store in column-major order
            h_attn_c_attn_w_Q_0[j * n_embd + i] = h_attn_c_attn_w_0[i * n_embd * 3 +              j];
            h_attn_c_attn_w_K_0[j * n_embd + i] = h_attn_c_attn_w_0[i * n_embd * 3 +     n_embd + j];
            h_attn_c_attn_w_V_0[j * n_embd + i] = h_attn_c_attn_w_0[i * n_embd * 3 + 2 * n_embd + j];
        }
        h_attn_c_attn_b_Q_0[i] = h_attn_c_attn_b_0[             i];
        h_attn_c_attn_b_K_0[i] = h_attn_c_attn_b_0[    n_embd + i];
        h_attn_c_attn_b_V_0[i] = h_attn_c_attn_b_0[2 * n_embd + i];
    }

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_w_Q_0, h_attn_c_attn_w_Q_0.data(), h_attn_c_attn_w_Q_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_w_K_0, h_attn_c_attn_w_K_0.data(), h_attn_c_attn_w_K_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_w_V_0, h_attn_c_attn_w_V_0.data(), h_attn_c_attn_w_V_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_b_Q_0, h_attn_c_attn_b_Q_0.data(), h_attn_c_attn_b_Q_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_b_K_0, h_attn_c_attn_b_K_0.data(), h_attn_c_attn_b_K_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn_c_attn_b_V_0, h_attn_c_attn_b_V_0.data(), h_attn_c_attn_b_V_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
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
