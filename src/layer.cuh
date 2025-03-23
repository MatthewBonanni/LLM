#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <string>
#include <vector>

#include <hdf5_hl.h>

#include "utils.cuh"

class Layer {
    public:
        Layer(uint32_t n_ctx, uint32_t n_embd, uint32_t n_head);
        ~Layer();
        void allocate_kv_cache(uint32_t batch_size);
        void apply(
            fp_t* d_hidden_states,
            fp_t* d_residual,
            fp_t* d_temp,
            uint32_t batch_size,
            uint32_t seq_length,
            uint32_t seq_offset);
        void load_from_hdf5(hid_t file_id, const std::string& layer_path);
        void copy_host_to_device();
        void copy_device_to_host();

    private:
        void launch_layer_normalization(
            fp_t* d_input,
            const fp_t* d_gamma,
            const fp_t* d_beta,
            uint32_t batch_size,
            uint32_t seq_length);
        
        void launch_qkv_projection(
            fp_t* d_hidden_states,
            fp_t* d_q,
            half* d_kv,
            fp_t* d_w_qkv,
            fp_t* d_b_qkv,
            uint32_t batch_size,
            uint32_t seq_length,
            uint32_t seq_offset);

        // Layer parameters
        uint32_t n_ctx;
        uint32_t n_embd;
        uint32_t n_head;

        // Attention
        std::vector<fp_t> h_attn_c_attn_w_0;
        std::vector<fp_t> h_attn_c_attn_b_0;
        std::vector<fp_t> h_attn_c_proj_w_0;
        std::vector<fp_t> h_attn_c_proj_b_0;
        fp_t* d_attn_c_attn_w_0;
        fp_t* d_attn_c_attn_b_0;
        fp_t* d_attn_c_proj_w_0;
        fp_t* d_attn_c_proj_b_0;

        // Layer normalization
        std::vector<fp_t> h_ln_1_b_0;
        std::vector<fp_t> h_ln_1_g_0;
        std::vector<fp_t> h_ln_2_b_0;
        std::vector<fp_t> h_ln_2_g_0;
        fp_t* d_ln_1_b_0;
        fp_t* d_ln_1_g_0;
        fp_t* d_ln_2_b_0;
        fp_t* d_ln_2_g_0;

        // MLP
        std::vector<fp_t> h_mlp_c_fc_w_0;
        std::vector<fp_t> h_mlp_c_fc_b_0;
        std::vector<fp_t> h_mlp_c_proj_w_0;
        std::vector<fp_t> h_mlp_c_proj_b_0;
        fp_t* d_mlp_c_fc_w_0;
        fp_t* d_mlp_c_fc_b_0;
        fp_t* d_mlp_c_proj_w_0;
        fp_t* d_mlp_c_proj_b_0;

        // KV cache
        half* d_kv_cache;
        uint32_t kv_cache_size;
};
