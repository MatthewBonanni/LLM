#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <hdf5_hl.h>

#include "utils.cuh"
#include "tokenizer.cuh"
#include "layer.cuh"

class LLM {
    public:
        LLM(const std::string& model_path, uint32_t batch_size);
        ~LLM();

        void print();
        void load_hparams(std::string model_path);
        void load_model(std::string model_path);
        void run_interactive();
        void tokenize_write_and_run_inference(const std::vector<std::string>& input_texts);
        void load_tokens_and_run_inference(const std::string& h5_file_path);
        void run_inference(
            const std::vector<id_t>& token_ids,
            uint32_t corpus_size,
            uint32_t seq_length);


    private:
        void allocate_temp_buffers(uint32_t seq_length);
        void free_temp_buffers();

        void tokenize(
            const std::vector<std::string>& input_texts,
            std::vector<id_t>& token_ids,
            uint32_t& corpus_size,
            uint32_t& seq_length);

        void write_token_ids(
            const std::string& h5_file_path,
            const std::vector<id_t>& token_ids,
            uint32_t corpus_size,
            uint32_t seq_length);

        void load_token_ids(
            const std::string& h5_file_path,
            std::vector<id_t>& token_ids,
            uint32_t& corpus_size,
            uint32_t& seq_length);

        void copy_params_host_to_device();

        void apply_embeddings(
            id_t* d_token_ids,
            fp_t* d_embeddings,
            uint32_t seq_length,
            uint32_t seq_offset);

        void initialize_kv_caches(uint32_t seq_length);

        void forward_pass(
            const std::vector<id_t>& token_ids,
            std::vector<fp_t>& logits,
            uint32_t seq_length,
            uint32_t seq_offset);

        void apply_final_layer_norm(
            fp_t* d_hidden_states,
            uint32_t seq_length);

        void apply_lm_head(
            fp_t* d_hidden_states,
            fp_t* d_logits,
            uint32_t seq_length);

        std::vector<std::pair<fp_t, id_t>> get_top_predictions(const std::vector<fp_t>& logits);

        std::vector<id_t> sample_tokens(const std::vector<std::pair<fp_t, id_t>>& probabilities);

        void append_new_tokens(
            std::vector<id_t>& generated_ids,
            const std::vector<id_t>& new_ids);

        bool all_eos(const std::vector<id_t>& ids);

        void generate_text_recursive(
            const std::vector<id_t>& input_ids,
            std::vector<id_t>& generated_ids,
            uint32_t seq_length);

        // Model hyperparameters
        uint32_t n_vocab;
        uint32_t n_ctx;
        uint32_t n_embd;
        uint32_t n_head;
        uint32_t n_layer;

        // Tokenizer
        Tokenizer tokenizer;
        std::vector<std::unique_ptr<Layer>> layers;

        // Model parameters
        std::vector<fp_t> h_wte_0;
        std::vector<fp_t> h_wpe_0;
        std::vector<fp_t> h_ln_f_b_0;
        std::vector<fp_t> h_ln_f_g_0;
        fp_t* d_wte_0;
        fp_t* d_wpe_0;
        fp_t* d_ln_f_b_0;
        fp_t* d_ln_f_g_0;

        // Computational parameters
        uint32_t batch_size;
        uint32_t max_out_length;
        fp_t temperature;
        uint32_t n_top_predictions;

        // Temporary buffers
        id_t* d_token_ids;
        fp_t* d_hidden_states;
        fp_t* d_residual;
        fp_t* d_temp;
        fp_t* d_logits;
};
