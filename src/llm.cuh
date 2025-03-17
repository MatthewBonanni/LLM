#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <hdf5_hl.h>

#include "tokenizer.cuh"
#include "layer.cuh"

class LLM {
    public:
        LLM(const std::string& model_path);
        ~LLM();

        void print();
        void load_hparams(std::string model_path);
        void load_model(std::string model_path);
        void run_interactive();

    private:
        void copy_params_host_to_device();
        void apply_embeddings(int* d_token_ids, float* d_embeddings, int token_count);
        std::vector<float> forward_pass(const std::vector<int>& h_token_ids);
        void apply_final_layer_norm(float* d_hidden_states, int seq_length);
        std::vector<std::pair<float, int>> get_top_predictions(const std::vector<float>& logits, int k);
        int sample_token(const std::vector<std::pair<float, int>>& probabilities, float temperature);
        void generate_text(const std::vector<int>& input_ids, int max_length, float temperature);
        void clean_up_memory(const std::vector<void*>& buffers);

        int n_vocab;
        int n_ctx;
        int n_embd;
        int n_head;
        int n_layer;

        Tokenizer tokenizer;

        std::vector<float> h_wte_0;
        std::vector<float> h_wpe_0;
        std::vector<float> h_ln_f_b_0;
        std::vector<float> h_ln_f_g_0;
        float* d_wte_0;
        float* d_wpe_0;
        float* d_ln_f_b_0;
        float* d_ln_f_g_0;

        std::vector<std::unique_ptr<Layer>> layers;
};
