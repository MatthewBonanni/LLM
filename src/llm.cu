#include "llm.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#include <nlohmann/json.hpp>
#include <hdf5_hl.h>

#include "utils.cuh"
#include "io.cuh"
#include "tokenizer.cuh"
#include "layer.cuh"
#include "kernels.cuh"

LLM::LLM(const std::string& model_path) :
        tokenizer(model_path),
        d_wte_0(nullptr),
        d_wpe_0(nullptr),
        d_ln_f_b_0(nullptr),
        d_ln_f_g_0(nullptr),
        max_out_length(50),
        temperature(0.7f),
        n_top_predictions(200) {
    load_hparams(model_path);
    load_model(model_path);

    if (n_top_predictions > n_vocab) {
        n_top_predictions = n_vocab;
    }
}

LLM::~LLM() {}

void LLM::print() {
    printf("--------------------------------\n");
    printf("LLM Configuration\n");
    printf("--------------------------------\n");
    printf("n_vocab: %d\n", n_vocab);
    printf("n_ctx: %d\n", n_ctx);
    printf("n_embd: %d\n", n_embd);
    printf("n_head: %d\n", n_head);
    printf("n_layer: %d\n", n_layer);
    printf("--------------------------------\n");
}

void LLM::load_hparams(std::string model_path) {
    std::ifstream file(model_path + "/hparams.json");
    if (!file) {
        std::cerr << "Error: Cannot open encoder.json" << std::endl;
        return;
    }

    nlohmann::json j;
    file >> j;
    n_vocab = j["n_vocab"];
    n_ctx = j["n_ctx"];
    n_embd = j["n_embd"];
    n_head = j["n_head"];
    n_layer = j["n_layer"];
}

void LLM::load_model(std::string model_path) {
    std::cout << "Loading model weights from " << model_path << std::endl;
    hid_t file_id = H5Fopen((model_path + "/model.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Error: Cannot open model.h5");
    }

    std::string base_path = "/model_weights/model";

    // Load embeddings
    std::cout << "> Embeddings" << std::endl;
    std::cout << "  > Allocating host and device memory..." << std::endl;
    h_wte_0.resize(n_vocab * n_embd);
    h_wpe_0.resize(n_ctx * n_embd);
    CHECK_CUDA(cudaMalloc(&d_wte_0, n_vocab * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wpe_0, n_ctx * n_embd * sizeof(float)));
    std::cout << "  > Loading weights..." << std::endl;
    read_dataset(file_id, base_path + "/wte_0", h_wte_0);
    read_dataset(file_id, base_path + "/wpe_0", h_wpe_0);

    // Load layers
    for (int i = 0; i < n_layer; i++) {
        std::cout << "> Layer " << i << std::endl;
        std::cout << "  > Allocating host and device memory..." << std::endl;
        layers.push_back(std::make_unique<Layer>(n_embd, n_head));

        std::string layer_path = base_path + "/h" + std::to_string(i);
        std::cout << "  > Loading weights..." << std::endl;
        layers[i]->load_from_hdf5(file_id, layer_path);
    }

    // Load final layer norm
    std::cout << "> Final layer norm" << std::endl;
    std::cout << "  > Allocating host and device memory..." << std::endl;
    h_ln_f_b_0.resize(n_embd);
    h_ln_f_g_0.resize(n_embd);
    CHECK_CUDA(cudaMalloc(&d_ln_f_b_0, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_f_g_0, n_embd * sizeof(float)));
    std::cout << "  > Loading weights..." << std::endl;
    read_dataset(file_id, base_path + "/ln_f/b_0", h_ln_f_b_0);
    read_dataset(file_id, base_path + "/ln_f/g_0", h_ln_f_g_0);

    // Close the file
    H5Fclose(file_id);

    // Copy weights to device
    copy_params_host_to_device();
}

void LLM::apply_embeddings(int* d_token_ids, float* d_embeddings, int batch_size, int seq_length) {
    // Each thread handles one element (i_batch, i_sequence, i_embedding)
    // in the embedding matrix (batch, sequence, embedding)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   (n_embd     + block_size.z - 1) / block_size.z);
    embedding_kernel<<<grid_size, block_size>>>(
        d_token_ids, d_wte_0, d_wpe_0, d_embeddings, batch_size, seq_length, n_embd);
}

void LLM::apply_final_layer_norm(float* d_hidden_states, int batch_size, int seq_length) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    layer_normalization_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_ln_f_g_0, d_ln_f_b_0, batch_size, seq_length, n_embd);
}

void LLM::apply_lm_head(float* d_hidden_state, float* d_logits, int batch_size, int seq_length) {
    // GPT-2 uses wte as the lm_head
    // Each thread handles one element (i_batch, i_vocab)
    // in the logits (batch, vocab)
    dim3 block_size(32, 32, 1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (n_vocab    + block_size.y - 1) / block_size.y,
                   1);
    lm_head_kernel<<<grid_size, block_size>>>(
        d_hidden_state, d_logits, d_wte_0, nullptr, batch_size, n_vocab, n_embd);
}

void LLM::copy_params_host_to_device() {
    CHECK_CUDA(cudaMemcpy(d_wte_0, h_wte_0.data(), h_wte_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wpe_0, h_wpe_0.data(), h_wpe_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < n_layer; i++) {
        layers[i]->copy_host_to_device();
    }
    CHECK_CUDA(cudaMemcpy(d_ln_f_b_0, h_ln_f_b_0.data(), h_ln_f_b_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_f_g_0, h_ln_f_g_0.data(), h_ln_f_g_0.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void LLM::run_interactive() {
    std::cout << "LLM Running Mode. Use CTRL-C to quit.\n";

    while (true) {
        // Get user input
        std::string input;
        std::cout << ">> ";
        std::getline(std::cin, input);

        // Tokenize input
        std::vector<int> h_token_ids = tokenizer.tokenize(input);

        // Print token info
        std::cout << "Token IDs: ";
        for (int id : h_token_ids) {
            std::cout << id << " ";
        }
        std::cout << "\nToken count: " << h_token_ids.size() << std::endl;

        // If empty input, continue
        if (h_token_ids.empty()) {
            std::cout << "Empty input, please try again.\n";
            continue;
        }

        // If input is too long, truncate
        if (h_token_ids.size() > n_ctx) {
            h_token_ids.resize(n_ctx);
            std::cout << "WARNING: Input too long, truncating to " << n_ctx << " tokens." << std::endl;
        }

        // Generate text
        // TODO: Give prior conversation as context
        std::cout << "Generated: ";
        std::vector<int> generated_ids;
        generate_text_recursive(h_token_ids, generated_ids, 1, h_token_ids.size());
        std::cout << std::endl;
    }
}

void LLM::run_inference(const std::vector<std::string>& input_texts) {
    // Tokenize inputs
    std::vector<std::vector<int>> token_batches;
    size_t max_seq_length = 0;
    
    std::cout << "Tokenizing input texts..." << std::endl;
    for (const auto& text : input_texts) {
        std::vector<int> tokens = tokenizer.tokenize(text);
        if (tokens.size() > n_ctx) {
            std::cout << "WARNING: Input too long, truncating to " << n_ctx << " tokens." << std::endl;
            tokens.resize(n_ctx);
        }
        max_seq_length = std::max(max_seq_length, tokens.size());
        token_batches.push_back(std::move(tokens));
    }
    
    // Pad sequences to the same length with EOS token
    for (auto& tokens : token_batches) {
        tokens.resize(max_seq_length, tokenizer.eos_token_id());
    }
    
    size_t batch_size = token_batches.size();
    std::vector<int> h_token_ids(batch_size * max_seq_length);
    
    // Flatten token_batches into h_token_ids
    for (size_t i = 0; i < batch_size; ++i) {
        std::copy(token_batches[i].begin(), token_batches[i].end(), h_token_ids.begin() + i * max_seq_length);
    }
    
    // Run inference
    std::cout << "Running inference on " << batch_size
              << " input texts of max length " << max_seq_length
              << "..." << std::endl;
    std::vector<int> generated_ids;
    generate_text_recursive(h_token_ids, generated_ids, batch_size, max_seq_length);
}


std::vector<float> LLM::forward_pass(const std::vector<int>& token_ids,
                                     int batch_size,
                                     int seq_length) {
    // Allocate device memory for token IDs and embeddings
    int* d_token_ids = nullptr;
    float* d_hidden_states = nullptr;
    float* d_residual = nullptr;
    float* d_temp = nullptr;
    float* d_logits = nullptr;
    std::vector<float> h_logits(n_vocab);

    int token_count = token_ids.size();
    if (token_ids.size() != batch_size * seq_length) {
        throw std::runtime_error("Error: token_ids.size() does not match batch_size * seq_length");
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_token_ids, token_count * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hidden_states, token_count * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_residual, token_count * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, token_count * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, batch_size * n_vocab * sizeof(float)));

    // Token IDs
    CHECK_CUDA(cudaMemcpy(d_token_ids, token_ids.data(), token_count * sizeof(int), cudaMemcpyHostToDevice));

    // Embeddings
    apply_embeddings(d_token_ids, d_hidden_states, batch_size, seq_length);

    // Process through transformer layers
    for (int i = 0; i < n_layer; i++) {
        layers[i]->apply(d_hidden_states, d_residual, d_temp, batch_size, seq_length);
    }

    // Apply final layer norm
    apply_final_layer_norm(d_hidden_states, batch_size, seq_length);

    // Get logits for the last token position
    apply_lm_head(d_hidden_states, d_logits, batch_size, seq_length);

    // Synchronize device
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy logits to host
    CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits, batch_size * n_vocab * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up resources
    clean_up_memory({d_token_ids, d_hidden_states, d_residual, d_logits});

    return h_logits;
}

std::vector<std::pair<float, int>> LLM::get_top_predictions(const std::vector<float>& logits,
                                                            int batch_size,
                                                            int seq_length) {
    std::vector<std::pair<float, int>> probs;
    probs.reserve(batch_size * n_vocab);

    // Scale logits by temperature
    std::vector<float> logits_temp = logits;
    for (auto &logit : logits_temp) {
        logit /= temperature;
    }

    for (int i = 0; i < batch_size; i++) {
        // Apply softmax for each batch
        float max_logit = *std::max_element(logits_temp.begin() + i * n_vocab,
                                            logits_temp.begin() + (i + 1) * n_vocab);
        float sum_exp = 0.0f;
        for (int j = 0; j < n_vocab; j++) {
            float prob = std::exp(logits_temp[i * n_vocab + j] - max_logit);
            sum_exp += prob;
            probs.push_back({prob, j});
        }
        for (int j = 0; j < n_vocab; j++) {
            probs[i * n_vocab + j].first /= sum_exp;
        }

        // Sort by probability (descending)
        std::partial_sort(probs.begin() + i * n_vocab,
                          probs.begin() + i * n_vocab + n_top_predictions,
                          probs.begin() + (i + 1) * n_vocab,
                          [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    return probs;
}

std::vector<int> LLM::sample_tokens(const std::vector<std::pair<float, int>>& probabilities,
                                    int batch_size,
                                    int seq_length) {
    // Sample based on adjusted probabilities
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<int> sampled_tokens(batch_size);
    for (int i = 0; i < batch_size; i++) {
        float r = dis(gen);
        float cdf = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            auto& p = probabilities[i * seq_length + j];
            cdf += p.first;
            if (r <= cdf) {
                sampled_tokens[i] = p.second;
                break;
            }
        }
    }
    return sampled_tokens;
}

void LLM::append_new_tokens(std::vector<int>& generated_ids,
                            std::vector<int>& context_ids,
                            const std::vector<int>& new_ids,
                            int batch_size,
                            int seq_length) {
    // Handle generated tokens (accumulating generated tokens)
    // Expand generated tokens
    int seq_length_generated = generated_ids.size() / batch_size;
    generated_ids.resize(batch_size * (seq_length_generated + 1));

    // Copy old generated tokens and add new token at the end
    // Work backwards to avoid overwriting
    for (int i = batch_size - 1; i >= 0; i--) {
        for (int j = seq_length_generated - 1; j >= 0; j--) {
            generated_ids[i * (seq_length_generated + 1) + j] = generated_ids[i * seq_length_generated + j];
        }
        generated_ids[i * (seq_length_generated + 1) + seq_length_generated] = new_ids[i];
    }

    // Handle context tokens (moving window)
    if (seq_length < n_ctx) {
        // Expand context tokens
        context_ids.resize(batch_size * (seq_length + 1));

        // Copy old context tokens and add new token at the end
        // Work backwards to avoid overwriting
        for (int i = batch_size - 1; i >= 0; i--) {
            for (int j = seq_length - 1; j >= 0; j--) {
                context_ids[i * (seq_length + 1) + j] = context_ids[i * seq_length + j];
            }
            context_ids[i * (seq_length + 1) + seq_length] = new_ids[i];
        }
    } else {
        // Shift context tokens to the left and add new token at the end
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < seq_length - 1; j++) {
                context_ids[i * seq_length + j] = context_ids[i * seq_length + j + 1];
            }
            context_ids[i * seq_length + seq_length - 1] = new_ids[i];
        }
    }
}

bool LLM::all_eos(const std::vector<int>& ids,
                  int batch_size,
                  int seq_length) {
    for (int i = 0; i < batch_size; i++) {
        if (ids[i * seq_length + seq_length - 1] != tokenizer.eos_token_id()) {
            return false;
        }
    }
    return true;
}

void LLM::generate_text_recursive(const std::vector<int>& input_ids,
                                  std::vector<int>& generated_ids,
                                  int batch_size,
                                  int seq_length) {
    std::flush(std::cout);
    std::vector<int> context_ids = input_ids;
    
    for (int gen_idx = 0; gen_idx < max_out_length; gen_idx++) {
        // Forward pass for the current sequence
        std::vector<float> logits = forward_pass(context_ids, batch_size, seq_length);
        
        // Get predictions
        std::vector<std::pair<float, int>> probabilities = get_top_predictions(logits, batch_size, seq_length);
        
        // Sample next token
        std::vector<int> next_ids = sample_tokens(probabilities, batch_size, seq_length);
        
        // Add to generated sequence
        append_new_tokens(generated_ids, context_ids, next_ids, batch_size, seq_length);
        
        // Print the token if batch size is 1
        if (batch_size == 1) {
            std::string token_str = tokenizer.detokenize({next_ids[0]});
            std::flush(std::cout);
        }

        // Check for EOS token
        if (all_eos(next_ids, batch_size, seq_length)) {
            break;
        }
    }
}

void LLM::clean_up_memory(const std::vector<void*>& buffers) {
    for (void* buffer : buffers) {
        if (buffer != nullptr) {
            CHECK_CUDA(cudaFree(buffer));
        }
    }
}