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

        int n_vocab;
        int n_ctx;
        int n_embd;
        int n_head;
        int n_layer;

        Tokenizer tokenizer;

        std::vector<float> h_wte;
        std::vector<float> h_wpe;
        float* d_wte;
        float* d_wpe;

        std::vector<Layer> layers;
};
