#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <hdf5_hl.h>

#include "utils.cuh"

class Layer {
    public:
        Layer(uint64_t n_embd, uint64_t n_head);
        ~Layer();
        void apply(
            fp_t* d_hidden_states,
            fp_t* d_residual,
            fp_t* d_temp,
            uint64_t batch_size,
            uint64_t seq_length);
        void load_from_hdf5(hid_t file_id, const std::string& layer_path);
        void copy_host_to_device();
        void copy_device_to_host();

    private:
        // Layer parameters
        uint64_t n_embd;
        uint64_t n_head;

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
};
