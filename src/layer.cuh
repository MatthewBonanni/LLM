#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <hdf5_hl.h>

class Layer {
    public:
        Layer(int n_embd, int n_head);
        ~Layer();
        void load_from_hdf5(hid_t file_id, const std::string& layer_path);
        void copy_host_to_device();
        void copy_device_to_host();

    private:
        // Attention
        std::vector<float> h_attn_c_attn_w_0;
        std::vector<float> h_attn_c_attn_b_0;
        std::vector<float> h_attn_c_proj_w_0;
        std::vector<float> h_attn_c_proj_b_0;
        float* d_attn_c_attn_w_0;
        float* d_attn_c_attn_b_0;
        float* d_attn_c_proj_w_0;
        float* d_attn_c_proj_b_0;

        // Layer normalization
        std::vector<float> h_ln_1_b_0;
        std::vector<float> h_ln_1_g_0;
        std::vector<float> h_ln_2_b_0;
        std::vector<float> h_ln_2_g_0;
        float* d_ln_1_b_0;
        float* d_ln_1_g_0;
        float* d_ln_2_b_0;
        float* d_ln_2_g_0;

        // MLP
        std::vector<float> h_mlp_c_fc_w_0;
        std::vector<float> h_mlp_c_fc_b_0;
        std::vector<float> h_mlp_c_proj_w_0;
        std::vector<float> h_mlp_c_proj_b_0;
        float* d_mlp_c_fc_w_0;
        float* d_mlp_c_fc_b_0;
        float* d_mlp_c_proj_w_0;
        float* d_mlp_c_proj_b_0;
};
