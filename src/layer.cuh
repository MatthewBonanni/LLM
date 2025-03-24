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

    private:
        void launch_layer_normalization(
            fp_t* d_input,
            uint32_t batch_size,
            uint32_t seq_length);
        
        void launch_qkv_projection(
            const fp_t* d_hidden_states,
            fp_t* d_q,
            uint32_t batch_size,
            uint32_t seq_length,
            uint32_t seq_offset);
        
        void launch_multi_head_attention(
            const fp_t* d_q,
            fp_t* d_output,
            uint32_t batch_size,
            uint32_t seq_length,
            uint32_t seq_offset);
        
        void launch_final_projection(
            const fp_t* d_input,
            fp_t* d_output,
            uint32_t batch_size,
            uint32_t seq_length);
        
        void launch_add_residual(
            const fp_t* d_input,
            const fp_t* d_residual,
            fp_t* d_output,
            uint32_t batch_size,
            uint32_t seq_length);
        
        void launch_mlp(
            const fp_t* d_input,
            fp_t* d_output,
            uint32_t batch_size,
            uint32_t seq_length);

        // Layer parameters
        uint32_t n_ctx;
        uint32_t n_embd;
        uint32_t n_head;

        // Attention
        fp_t* d_attn_c_attn_w_Q_0;
        fp_t* d_attn_c_attn_w_K_0;
        fp_t* d_attn_c_attn_w_V_0;
        fp_t* d_attn_c_attn_b_Q_0;
        fp_t* d_attn_c_attn_b_K_0;
        fp_t* d_attn_c_attn_b_V_0;
        fp_t* d_attn_c_proj_w_0;
        fp_t* d_attn_c_proj_b_0;

        // Layer normalization
        fp_t* d_ln_1_b_0;
        fp_t* d_ln_1_g_0;
        fp_t* d_ln_2_b_0;
        fp_t* d_ln_2_g_0;

        // MLP
        fp_t* d_mlp_c_fc_w_0;
        fp_t* d_mlp_c_fc_b_0;
        fp_t* d_mlp_c_proj_w_0;
        fp_t* d_mlp_c_proj_b_0;

        // KV cache
        half* d_k_cache;
        half* d_v_cache;
        uint32_t kv_cache_size;
};
