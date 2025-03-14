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
        d_wte(nullptr),
        d_wpe(nullptr) {
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
    h_wte.resize(n_vocab * n_embd);
    h_wpe.resize(n_ctx * n_embd);
    CHECK_CUDA(cudaMalloc(&d_wte, n_vocab * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wpe, n_ctx * n_embd * sizeof(float)));
    std::cout << "  > Loading weights..." << std::endl;
    read_dataset(file_id, base_path + "/wte_0", h_wte);
    read_dataset(file_id, base_path + "/wpe_0", h_wpe);

    // Load layers
    for (int i = 0; i < n_layer; i++) {
        std::cout << "> Layer " << i << std::endl;
        std::cout << "  > Allocating host and device memory..." << std::endl;
        layers.push_back(std::make_unique<Layer>(n_embd, n_head));

        std::string layer_path = base_path + "/h" + std::to_string(i);
        std::cout << "  > Loading weights..." << std::endl;
        layers[i]->load_from_hdf5(file_id, layer_path);
    }

    // Close the file
    H5Fclose(file_id);

    // Copy weights to device
    copy_params_host_to_device();
}

void LLM::run_interactive() {
    std::cout << "LLM Running Mode. Use CTRL-C to quit.\n";

    while (true) {
        std::string input;
        std::cout << ">> ";
        std::getline(std::cin, input);

        std::vector<int> token_ids = tokenizer.tokenize(input);

        std::cout << "Token IDs: ";
        for (int i = 0; i < token_ids.size(); i++) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Token count: " << token_ids.size() << std::endl;
        
        // std::vector<int> output_ids = generate(input_ids);
        // std::vector<std::string> output_tokens = tokenizer.convert_ids_to_tokens(output_ids);
        // std::string output = tokenizer.detokenize(output_tokens);
    }
}

void LLM::copy_params_host_to_device() {
    CHECK_CUDA(cudaMemcpy(d_wte, h_wte.data(), h_wte.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wpe, h_wpe.data(), h_wpe.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < n_layer; i++) {
        layers[i]->copy_host_to_device();
    }
}
