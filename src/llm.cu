#include "llm.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#include <nlohmann/json.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "utils.cuh"
#include "io.cuh"
#include "tokenizer.cuh"
#include "layer.cuh"
#include "kernels.cuh"

LLM::LLM(const std::string& model_path,
         uint32_t batch_size) :
        tokenizer(model_path),
        d_wte_0(nullptr),
        d_wpe_0(nullptr),
        d_ln_f_b_0(nullptr),
        d_ln_f_g_0(nullptr),
        max_out_length(50),
        temperature(0.7f),
        n_top_predictions(200),
        batch_size(batch_size),
        d_token_ids(nullptr),
        d_hidden_states(nullptr),
        d_residual(nullptr),
        d_temp(nullptr),
        d_logits(nullptr) {
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
    printf("Model hyperparameters:\n");
    printf("> n_vocab: %d\n", n_vocab);
    printf("> n_ctx: %d\n", n_ctx);
    printf("> n_embd: %d\n", n_embd);
    printf("> n_head: %d\n", n_head);
    printf("> n_layer: %d\n", n_layer);
    printf("Runtime parameters:\n");
    printf("> max_out_length: %d\n", max_out_length);
    printf("> temperature: %.2f\n", temperature);
    printf("> n_top_predictions: %d\n", n_top_predictions);
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
    CHECK_CUDA(cudaMalloc(&d_wte_0, n_vocab * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_wpe_0, n_ctx * n_embd * sizeof(fp_t)));
    std::cout << "  > Loading weights..." << std::endl;
    read_dataset(file_id, base_path + "/wte_0", h_wte_0);
    read_dataset(file_id, base_path + "/wpe_0", h_wpe_0);

    // Load layers
    for (uint32_t i = 0; i < n_layer; i++) {
        std::cout << "> Layer " << i << std::endl;
        std::cout << "  > Allocating host and device memory..." << std::endl;
        layers.push_back(std::make_unique<Layer>(n_ctx, n_embd, n_head));

        std::string layer_path = base_path + "/h" + std::to_string(i);
        std::cout << "  > Loading weights..." << std::endl;
        layers[i]->load_from_hdf5(file_id, layer_path);
    }

    // Load final layer norm
    std::cout << "> Final layer norm" << std::endl;
    std::cout << "  > Allocating host and device memory..." << std::endl;
    h_ln_f_b_0.resize(n_embd);
    h_ln_f_g_0.resize(n_embd);
    CHECK_CUDA(cudaMalloc(&d_ln_f_b_0, n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_ln_f_g_0, n_embd * sizeof(fp_t)));
    std::cout << "  > Loading weights..." << std::endl;
    read_dataset(file_id, base_path + "/ln_f/b_0", h_ln_f_b_0);
    read_dataset(file_id, base_path + "/ln_f/g_0", h_ln_f_g_0);

    // Close the file
    H5Fclose(file_id);

    // Copy weights to device
    copy_params_host_to_device();
}

void LLM::allocate_temp_buffers(uint32_t seq_length) {
    // Allocate temporary buffers
    CHECK_CUDA(cudaMalloc(&d_token_ids,     batch_size * seq_length * sizeof(id_t)));
    CHECK_CUDA(cudaMalloc(&d_hidden_states, batch_size * seq_length * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_residual,      batch_size * seq_length * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_temp,          batch_size * seq_length * n_embd * sizeof(fp_t)));
    CHECK_CUDA(cudaMalloc(&d_logits,        batch_size * n_vocab * sizeof(fp_t)));
}

void LLM::free_temp_buffers() {
    std::vector<void*> buffers = {
        d_token_ids,
        d_hidden_states,
        d_residual,
        d_temp,
        d_logits
    };
    clean_up_memory(buffers);
}

void LLM::apply_embeddings(id_t* d_token_ids,
                           fp_t* d_embeddings,
                           uint32_t seq_length,
                           uint32_t seq_offset) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the token_ids (batch, sequence, embedding)
    dim3 block_size(std::min(batch_size, (uint32_t)32),
                    std::min(seq_length, (uint32_t)32),
                    1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    embedding_kernel<<<grid_size, block_size>>>(
        d_token_ids, d_wte_0, d_wpe_0, d_embeddings,
        batch_size, seq_length, seq_offset, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void LLM::apply_final_layer_norm(fp_t* d_hidden_states,
                                 uint32_t seq_length) {
    // Each thread handles one token (i_batch, i_sequence, :)
    // in the hidden states (batch, sequence, embedding)
    dim3 block_size(std::min(batch_size, (uint32_t)32),
                    std::min(seq_length, (uint32_t)32),
                    1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (seq_length + block_size.y - 1) / block_size.y,
                   1);
    layer_normalization_kernel<<<grid_size, block_size>>>(
        d_hidden_states, d_ln_f_g_0, d_ln_f_b_0,
        batch_size, seq_length, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void LLM::apply_lm_head(fp_t* d_hidden_state,
                        fp_t* d_logits,
                        uint32_t seq_length) {
    // GPT-2 uses wte as the lm_head
    // Each thread handles one element (i_batch, i_vocab)
    // in the logits (batch, vocab)
    dim3 block_size(std::min(batch_size, (uint32_t)32),
                    std::min(n_vocab,    (uint32_t)32),
                    1);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                   (n_vocab    + block_size.y - 1) / block_size.y,
                   1);
    lm_head_kernel<<<grid_size, block_size>>>(
        d_hidden_state, d_logits, d_wte_0, nullptr,
        batch_size, seq_length, n_vocab, n_embd);
    CHECK_CUDA(cudaGetLastError());
}

void LLM::copy_params_host_to_device() {
    CHECK_CUDA(cudaMemcpy(d_wte_0, h_wte_0.data(), h_wte_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wpe_0, h_wpe_0.data(), h_wpe_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    for (uint32_t i = 0; i < n_layer; i++) {
        layers[i]->copy_host_to_device();
    }
    CHECK_CUDA(cudaMemcpy(d_ln_f_b_0, h_ln_f_b_0.data(), h_ln_f_b_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ln_f_g_0, h_ln_f_g_0.data(), h_ln_f_g_0.size() * sizeof(fp_t), cudaMemcpyHostToDevice));
}

void LLM::run_interactive() {
    batch_size = 1;  // Set batch size to 1 for interactive mode

    std::cout << "Allocating temporary buffers..." << std::endl;
    allocate_temp_buffers(n_ctx);
    for (auto& layer : layers) {
        layer->allocate_kv_cache(1);
    }

    std::cout << "LLM Running Mode. Use CTRL-C to quit.\n";

    while (true) {
        // Get user input
        std::string input;
        std::cout << ">> ";
        std::getline(std::cin, input);

        // Tokenize input
        std::vector<id_t> token_ids = tokenizer.tokenize(input);

        // Print token info
        std::cout << "Token IDs: ";
        for (auto id : token_ids) {
            std::cout << id << " ";
        }
        std::cout << "\nToken count: " << token_ids.size() << std::endl;

        // If empty input, continue
        if (token_ids.empty()) {
            std::cout << "Empty input, please try again.\n";
            continue;
        }

        // If input is too long, truncate
        if (token_ids.size() > n_ctx) {
            token_ids.resize(n_ctx);
            std::cout << "WARNING: Input too long, truncating to " << n_ctx << " tokens." << std::endl;
        }

        // Generate text
        // TODO: Give prior conversation as context
        std::cout << "Generated: ";
        std::flush(std::cout);
        std::vector<id_t> generated_ids;
        generate_text_recursive(token_ids, generated_ids, token_ids.size());

        // Print generated tokens
        for (auto id : generated_ids) {
            std::string token_str = tokenizer.detokenize({id});
            std::cout << token_str << " ";
        }
        std::cout << std::endl;
    }

    free_temp_buffers();
}

void LLM::tokenize(const std::vector<std::string>& input_texts,
                   std::vector<id_t>& token_ids,
                   uint32_t& corpus_size,
                   uint32_t& seq_length) {
    // Tokenize inputs
    seq_length = 0;
    std::vector<std::vector<id_t>> token_ids_seqs;
    std::cout << "Tokenizing input texts..." << std::endl;
    for (const auto& text : input_texts) {
        std::vector<id_t> token_ids_i = tokenizer.tokenize(text);
        if (token_ids_i.size() > n_ctx) {
            std::cout << "WARNING: Input too long, truncating to " << n_ctx << " tokens." << std::endl;
            token_ids_i.resize(n_ctx);
        }
        seq_length = std::max(seq_length, (uint32_t)token_ids_i.size());
        token_ids_seqs.push_back(std::move(token_ids_i));
    }

    // Set corpus size
    corpus_size = token_ids_seqs.size();

    // Resize token_ids to hold all batches
    // Use EOS token as padding
    token_ids.resize(corpus_size * seq_length);
    std::fill(token_ids.begin(), token_ids.end(), tokenizer.eos_token_id());

    // Write token sequences into token_ids, right-aligned
    for (uint32_t i = 0; i < batch_size; ++i) {
        std::copy(token_ids_seqs[i].begin(),
                  token_ids_seqs[i].end(),
                  token_ids.begin() + i * seq_length + (seq_length - token_ids_seqs[i].size()));
    }
}

void LLM::write_token_ids(const std::string& h5_file_path,
                          const std::vector<id_t>& token_ids,
                          uint32_t corpus_size,
                          uint32_t seq_length) {
    // Write token IDs to H5 file
    hid_t file_id = H5Fcreate(h5_file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Error: Cannot create H5 file");
    }
    
    // Create dataset for token IDs
    hsize_t dims[1] = {token_ids.size()};
    hid_t dataspace_id = H5Screate_simple(1, dims, nullptr);
    hid_t dataset_id = H5Dcreate(file_id, "/token_ids", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Write token IDs data
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, token_ids.data());
    
    // Create attribute space (scalar)
    hid_t attr_space = H5Screate(H5S_SCALAR);
    
    // Create and write corpus_size attribute
    hid_t attr_id_corpus = H5Acreate(dataset_id, "corpus_size", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id_corpus, H5T_NATIVE_INT, &corpus_size);
    H5Aclose(attr_id_corpus);
    
    // Create and write seq_length attribute
    hid_t attr_id_seq = H5Acreate(dataset_id, "seq_length", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id_seq, H5T_NATIVE_INT, &seq_length);
    H5Aclose(attr_id_seq);
    
    // Close resources
    H5Sclose(attr_space);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
}

void LLM::load_token_ids(const std::string& h5_file_path,
                         std::vector<id_t>& token_ids,
                         uint32_t& corpus_size,
                         uint32_t& seq_length) {
    // Load token IDs from H5 file
    hid_t file_id = H5Fopen(h5_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Error: Cannot open H5 file");
    }
    
    // Open the dataset
    hid_t dataset_id = H5Dopen(file_id, "/token_ids", H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("Error: Cannot open dataset");
    }
    
    // Get the dataspace to determine the size
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    
    // Resize the vector to hold the data
    token_ids.resize(dims[0]);
    
    // Read token IDs data
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, token_ids.data());
    
    // Read corpus_size attribute
    int32_t corpus_size_attr;
    hid_t attr_id_corpus = H5Aopen(dataset_id, "corpus_size", H5P_DEFAULT);
    H5Aread(attr_id_corpus, H5T_NATIVE_INT, &corpus_size_attr);
    H5Aclose(attr_id_corpus);
    corpus_size = corpus_size_attr;

    // Read seq_length attribute
    int32_t seq_length_attr;
    hid_t attr_id_seq = H5Aopen(dataset_id, "seq_length", H5P_DEFAULT);
    H5Aread(attr_id_seq, H5T_NATIVE_INT, &seq_length_attr);
    H5Aclose(attr_id_seq);
    seq_length = seq_length_attr;

    // Close resources
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

void LLM::tokenize_write_and_run_inference(const std::vector<std::string>& input_texts) {
    // Tokenize input
    uint32_t corpus_size, seq_length;
    std::vector<id_t> token_ids;
    tokenize(input_texts, token_ids, corpus_size, seq_length);

    // Write token IDs to H5 file
    std::string h5_file_path = "token_ids.h5";
    write_token_ids(h5_file_path, token_ids, corpus_size, seq_length);

    // Run inference
    run_inference(token_ids, corpus_size, seq_length);
}

void LLM::load_tokens_and_run_inference(const std::string& h5_file_path) {
    // Load token IDs from H5 file
    uint32_t corpus_size, seq_length;
    std::vector<id_t> token_ids;
    load_token_ids(h5_file_path, token_ids, corpus_size, seq_length);

    // Run inference
    run_inference(token_ids, corpus_size, seq_length);
}

void LLM::run_inference(const std::vector<id_t>& token_ids,
                        uint32_t corpus_size,
                        uint32_t seq_length) {
    std::vector<id_t> token_ids_adjusted = token_ids;
    if (corpus_size > batch_size) {
        std::cout << "WARNING: corpus_size > batch_size, only running on the first batch." << std::endl;
        token_ids_adjusted.resize(batch_size * seq_length);
    } else {
        batch_size = corpus_size;
    }

    std::cout << "Allocating temporary buffers..." << std::endl;
    allocate_temp_buffers(seq_length);
    for (auto& layer : layers) {
        layer->allocate_kv_cache(batch_size);
    }

    std::cout << "Running inference on 1 batch of " << batch_size
              << " input sequences of length " << seq_length
              << "..." << std::endl;
    std::vector<id_t> generated_ids;
    generate_text_recursive(token_ids_adjusted, generated_ids, seq_length);

    std::cout << "Done. Freeing temporary buffers..." << std::endl;
    free_temp_buffers();
}

void LLM::forward_pass(const std::vector<id_t>& token_ids,
                       std::vector<fp_t>& logits,
                       uint32_t seq_length,
                       uint32_t seq_offset) {
    // Token IDs
    CHECK_CUDA(cudaMemcpy(
        d_token_ids,
        token_ids.data(),
        (uint64_t)batch_size * seq_length * sizeof(id_t),
        cudaMemcpyHostToDevice));

    // Embeddings
    apply_embeddings(d_token_ids, d_hidden_states, seq_length, seq_offset);

    // Process through transformer layers
    for (uint32_t i = 0; i < n_layer; i++) {
        layers[i]->apply(
            d_hidden_states,
            d_residual,
            d_temp,
            batch_size,
            seq_length,
            seq_offset);
    }

    // Apply final layer norm
    apply_final_layer_norm(d_hidden_states, seq_length);

    // Get logits for the last token position
    apply_lm_head(d_hidden_states, d_logits, seq_length);

    // Synchronize device
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy logits to host
    CHECK_CUDA(cudaMemcpy(
        logits.data(),
        d_logits,
        (uint64_t)batch_size * n_vocab * sizeof(fp_t),
        cudaMemcpyDeviceToHost));
}

std::vector<std::pair<fp_t, id_t>> LLM::get_top_predictions(const std::vector<fp_t>& logits) {
    std::vector<std::pair<fp_t, id_t>> probs;
    probs.reserve(batch_size * n_vocab);

    // Scale logits by temperature
    std::vector<fp_t> logits_temp = logits;
    for (auto &logit : logits_temp) {
        logit /= temperature;
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        // Apply softmax for each batch
        fp_t max_logit = *std::max_element(logits_temp.begin() + i * n_vocab,
                                           logits_temp.begin() + (i + 1) * n_vocab);
        fp_t sum_exp = 0.0f;
        for (uint32_t j = 0; j < n_vocab; j++) {
            fp_t prob = std::exp(logits_temp[(uint64_t)i * n_vocab + j] - max_logit);
            sum_exp += prob;
            probs.push_back({prob, j});
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            probs[(uint64_t)i * n_vocab + j].first /= sum_exp;
        }

        // Sort by probability (descending)
        std::partial_sort(probs.begin() + i * n_vocab,
                          probs.begin() + i * n_vocab + n_top_predictions,
                          probs.begin() + (i + 1) * n_vocab,
                          [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    return probs;
}

std::vector<id_t> LLM::sample_tokens(const std::vector<std::pair<fp_t, id_t>>& probabilities) {
    // Sample based on adjusted probabilities
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<fp_t> dis(0.0f, 1.0f);
    std::vector<id_t> sampled_tokens(batch_size);
    for (uint32_t i = 0; i < batch_size; i++) {
        fp_t r = dis(gen);
        fp_t cdf = 0.0f;
        for (uint32_t j = 0; j < n_vocab; j++) {
            auto& p = probabilities[(uint64_t)i * n_vocab + j];
            cdf += p.first;
            if (r <= cdf) {
                sampled_tokens[i] = p.second;
                break;
            }
        }
    }
    return sampled_tokens;
}

void LLM::append_new_tokens(std::vector<id_t>& generated_ids,
                            const std::vector<id_t>& new_ids) {
    // Handle generated tokens (accumulating generated tokens)
    // Expand generated tokens
    uint32_t seq_length_generated = generated_ids.size() / batch_size;
    generated_ids.resize(batch_size * (seq_length_generated + 1));

    // Copy old generated tokens and add new token at the end
    // Work backwards to avoid overwriting
    for (int32_t i = batch_size - 1; i >= 0; i--) {
        for (int32_t j = seq_length_generated - 1; j >= 0; j--) {
            generated_ids[(int64_t)i * (seq_length_generated + 1) + j] =
                generated_ids[(int64_t)i * seq_length_generated + j];
        }
        generated_ids[(int64_t)i * (seq_length_generated + 1) + seq_length_generated] = new_ids[i];
    }
}

bool LLM::all_eos(const std::vector<id_t>& ids) {
    for (uint32_t i = 0; i < batch_size; i++) {
        if (ids[i] != tokenizer.eos_token_id()) {
            return false;
        }
    }
    return true;
}

void LLM::generate_text_recursive(const std::vector<id_t>& input_ids,
                                  std::vector<id_t>& generated_ids,
                                  uint32_t seq_length) {
    if (input_ids.size() != batch_size * seq_length) {
        throw std::runtime_error("Error: input_ids size does not match batch_size * seq_length");
    }
    std::vector<id_t> token_ids = input_ids;
    std::vector<fp_t> logits(batch_size * n_vocab);

    uint32_t seq_offset = 0;
    for (uint32_t i_gen = 0; i_gen < max_out_length; i_gen++) {
        // Check to make sure n_ctx is not exceeded
        if (seq_offset + seq_length >= n_ctx) {
            throw std::runtime_error("Error: n_ctx exceeded");
        }

        // Forward pass for the current sequence
        forward_pass(token_ids, logits, seq_length, seq_offset);
        
        // Get predictions
        std::vector<std::pair<fp_t, id_t>> probabilities = get_top_predictions(logits);
        
        // Sample next token
        token_ids = sample_tokens(probabilities);

        // Add to generated sequence
        append_new_tokens(generated_ids, token_ids);

        // Print the token if batch size is 1
        if (batch_size == 1) {
            std::string token_str = tokenizer.detokenize({token_ids[0]});
            std::cout << token_str;
            std::flush(std::cout);
        } else {
            std::cout << "Finished generating " << i_gen + 1 << " new sets of tokens." << std::endl;
        }

        // Check for EOS token
        if (all_eos(token_ids)) {
            break;
        }

        // Sequence length is 1 after the first iteration
        seq_offset += seq_length;
        seq_length = 1;
    }
}