#include "llm.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <hdf5_hl.h>

#include "utils.cuh"
#include "io.cuh"
#include "tokenizer.cuh"
#include "layer.cuh"

LLM::LLM(const std::string& model_path) :
        tokenizer(model_path),
        d_wte_0(nullptr),
        d_wpe_0(nullptr),
        d_ln_f_b_0(nullptr),
        d_ln_f_g_0(nullptr) {
    load_hparams(model_path);
    load_model(model_path);
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

__global__ void embedding_kernel(const int* token_ids,
                                 const float* wte,
                                 const float* wpe,
                                 float* embeddings,
                                 int token_count,
                                 int embedding_dim) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process an element
    if (idx < token_count * embedding_dim) {
        // Calculate which token and which embedding dimension this thread is handling
        int token_idx = idx / embedding_dim;    // Which token
        int embd_idx = idx % embedding_dim;     // Which dimension in the embedding
        
        // Get the token ID for this position
        int token_id = token_ids[token_idx];
        
        // Calculate offset in embedding tables
        int token_offset = token_id * embedding_dim + embd_idx;
        int pos_offset = token_idx * embedding_dim + embd_idx;
        
        // Sum token embedding and positional embedding
        embeddings[idx] = wte[token_offset] + wpe[pos_offset];
    }
}

void LLM::apply_embeddings(int* d_token_ids, float* d_embeddings, int token_count) {
    // Kernel to compute final embeddings
    int threads = 256;
    int blocks = (token_count * n_embd + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(d_token_ids, d_wte_0, d_wpe_0, d_embeddings, token_count, n_embd);
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void layer_norm_kernel(float* hidden_states, const float* gamma, const float* beta,
                                  int seq_length, int hidden_size, float epsilon) {
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sum_sq = shared_data + blockDim.x;
    
    int pos = blockIdx.x;
    int tid = threadIdx.x;
    float* pos_hidden = hidden_states + pos * hidden_size;
    
    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    shared_sum_sq[tid] = 0.0f;
    
    // Calculate partial sums for mean and variance
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = pos_hidden[i];
        shared_sum[tid] += val;
        shared_sum_sq[tid] += val * val;
    }
    __syncthreads();
    
    // Parallel reduction for sum and sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Calculate mean and variance
    float mean = shared_sum[0] / hidden_size;
    float var = (shared_sum_sq[0] / hidden_size) - (mean * mean) + epsilon;
    float inv_std = rsqrtf(var);
    
    // Apply normalization with gamma and beta
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (pos_hidden[i] - mean) * inv_std;
        pos_hidden[i] = gamma[i] * normalized + beta[i];
    }
}

void LLM::apply_final_layer_norm(float* d_hidden_states, int seq_length) {
    // Layer norm parameters
    const float epsilon = 1e-5f;
    
    // Launch one block per sequence position, with threads for hidden dimension
    dim3 grid(seq_length);
    dim3 block(256);
    size_t shared_mem_size = 2 * block.x * sizeof(float);
    
    layer_norm_kernel<<<grid, block, shared_mem_size>>>(
        d_hidden_states, d_ln_f_g_0, d_ln_f_b_0, seq_length, n_embd, epsilon);
}

void LLM::copy_params_host_to_device() {
    CHECK_CUDA(cudaMemcpy(d_wte_0, h_wte_0.data(), h_wte_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wpe_0, h_wpe_0.data(), h_wpe_0.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < n_layer; i++) {
        layers[i]->copy_host_to_device();
    }
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
        
        // Generate text
        generate_text(h_token_ids, 50, 0.8f);
    }
}

std::vector<float> LLM::forward_pass(const std::vector<int>& tokens) {
    // Allocate device memory for token IDs and embeddings
    int* d_token_ids = nullptr;
    float* d_embeddings = nullptr;
    float* d_hidden_states = nullptr;
    float* d_residual = nullptr;
    float* d_logits = nullptr;
    std::vector<float> h_logits(n_vocab);
    
    // Token IDs
    CHECK_CUDA(cudaMalloc(&d_token_ids, tokens.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_token_ids, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
        
    // Embeddings
    CHECK_CUDA(cudaMalloc(&d_embeddings, tokens.size() * n_embd * sizeof(float)));
    apply_embeddings(d_token_ids, d_embeddings, tokens.size());
    CHECK_CUDA(cudaFree(d_token_ids)); // Free token IDs after embedding
    d_token_ids = nullptr;
        
    // Hidden states and residual
    CHECK_CUDA(cudaMalloc(&d_hidden_states, tokens.size() * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_residual, tokens.size() * n_embd * sizeof(float)));
        
    // Copy embeddings to hidden states
    CHECK_CUDA(cudaMemcpy(d_hidden_states, d_embeddings, tokens.size() * n_embd * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(d_embeddings)); // Free embeddings after copying
    d_embeddings = nullptr;
        
    // Process through transformer layers
    for (int i = 0; i < n_layer; i++) {
        layers[i]->apply(d_hidden_states, d_residual, tokens.size());
    }
        
    // Apply final layer norm
    apply_final_layer_norm(d_hidden_states, tokens.size());
        
    // Get logits for the last token position
    CHECK_CUDA(cudaMalloc(&d_logits, n_vocab * sizeof(float)));
    apply_lm_head(d_hidden_states + (tokens.size() - 1) * n_embd, d_logits);
        
    // Copy logits to host
    CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits, n_vocab * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up resources
    clean_up_memory({d_token_ids, d_embeddings, d_hidden_states, d_residual, d_logits});
    
    return h_logits;
}

std::vector<std::pair<float, int>> LLM::get_top_predictions(const std::vector<float>& logits, int k) {
    std::vector<std::pair<float, int>> probs;
    probs.reserve(n_vocab);
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    
    // Apply softmax
    for (int i = 0; i < n_vocab; i++) {
        float prob = std::exp(logits[i] - max_logit);
        sum_exp += prob;
        probs.push_back({prob, i});
    }
    
    // Normalize
    for (auto& p : probs) {
        p.first /= sum_exp;
    }
    
    // Sort by probability (descending)
    std::partial_sort(probs.begin(), probs.begin() + k, probs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    return probs;
}

int LLM::sample_token(const std::vector<std::pair<float, int>>& probs, float temperature) {
    // If temperature is 0, do greedy sampling
    if (temperature == 0.0f) {
        return probs[0].second;
    }
    
    // Create a copy for temperature adjustment
    std::vector<std::pair<float, int>> temp_adjusted = probs;
    
    // Apply temperature adjustment
    if (temperature != 1.0f) {
        float sum = 0.0f;
        for (auto& p : temp_adjusted) {
            p.first = std::pow(p.first, 1.0f / temperature);
            sum += p.first;
        }
        // Renormalize
        for (auto& p : temp_adjusted) {
            p.first /= sum;
        }
    }
    
    // Sample based on adjusted probabilities
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cdf = 0.0f;
    
    for (const auto& p : temp_adjusted) {
        cdf += p.first;
        if (r <= cdf) {
            return p.second;
        }
    }
    
    // Fallback to most likely token
    return temp_adjusted[0].second;
}

void LLM::generate_text(const std::vector<int>& input_ids, int max_tokens, float temperature) {
    std::cout << "Generated: ";
    std::vector<int> generated_tokens = input_ids;
    
    for (int gen_idx = 0; gen_idx < max_tokens; gen_idx++) {
        // Forward pass for the current sequence
        std::vector<float> logits = forward_pass(generated_tokens);
        
        // Get predictions
        auto predictions = get_top_predictions(logits, 40);
        
        // Sample next token
        int next_token = sample_token(predictions, temperature);
        
        // Add to generated sequence
        generated_tokens.push_back(next_token);
        
        // Print the token
        std::string token_str = tokenizer.detokenize({next_token});
        std::cout << token_str;
        std::cout.flush();
        
        // Check for EOS token
        if (next_token == tokenizer.eos_token_id()) {
            break;
        }
    }
    
    std::cout << std::endl;
}

void LLM::clean_up_memory(const std::vector<void*>& buffers) {
    for (void* buffer : buffers) {
        if (buffer != nullptr) {
            CHECK_CUDA(cudaFree(buffer));
        }
    }
}