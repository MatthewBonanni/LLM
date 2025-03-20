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
        void tokenize_write_and_run_inference(const std::vector<std::string>& input_texts);
        void load_tokens_and_run_inference(const std::string& h5_file_path);
        void run_inference(const std::vector<int>& token_ids,
                           int batch_size,
                           int seq_length);


    private:
        void tokenize(const std::vector<std::string>& input_texts,
                      std::vector<int>& token_ids,
                      int& batch_size,
                      int& seq_length);
        void write_token_ids(const std::string& h5_file_path,
                             const std::vector<int>& token_ids,
                             int batch_size,
                             int seq_length);
        void load_token_ids(const std::string& h5_file_path,
                            std::vector<int>& token_ids,
                            int& batch_size,
                            int& seq_length);
        void copy_params_host_to_device();
        void apply_embeddings(int* d_token_ids,
                              float* d_embeddings,
                              int batch_size,
                              int seq_length);
        std::vector<float> forward_pass(const std::vector<int>& token_ids,
                                        int batch_size,
                                        int seq_length);
        void apply_final_layer_norm(float* d_hidden_states,
                                    int batch_size,
                                    int seq_length);
        void apply_lm_head(float* d_hidden_states,
                           float* d_logits,
                           int batch_size,
                           int seq_length);
        std::vector<std::pair<float, int>> get_top_predictions(const std::vector<float>& logits,
                                                               int batch_size,
                                                               int seq_length);
        std::vector<int> sample_tokens(const std::vector<std::pair<float, int>>& probabilities,
                                       int batch_size,
                                       int seq_length);
        void append_new_tokens(std::vector<int>& generated_ids,
                               std::vector<int>& context_ids,
                               const std::vector<int>& new_ids,
                               int batch_size,
                               int& seq_length);
        bool all_eos(const std::vector<int>& ids,
                     int batch_size,
                     int seq_length);
        void generate_text_recursive(const std::vector<int>& input_ids,
                                     std::vector<int>& generated_ids,
                                     int batch_size,
                                     int& seq_length);
        void clean_up_memory(const std::vector<void*>& buffers);

        // Model hyperparameters
        int n_vocab;
        int n_ctx;
        int n_embd;
        int n_head;
        int n_layer;

        // Tokenizer
        Tokenizer tokenizer;
        std::vector<std::unique_ptr<Layer>> layers;

        // Model parameters
        std::vector<float> h_wte_0;
        std::vector<float> h_wpe_0;
        std::vector<float> h_ln_f_b_0;
        std::vector<float> h_ln_f_g_0;
        float* d_wte_0;
        float* d_wpe_0;
        float* d_ln_f_b_0;
        float* d_ln_f_g_0;

        // Computational parameters
        int max_out_length;
        float temperature;
        int n_top_predictions;
};
