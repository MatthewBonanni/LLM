#pragma once

#include <cuda_runtime.h>

#define INTERMEDIATE_SIZE_MAX 4096
#define SEQ_LENGTH_MAX 2048

__device__ __host__ float gelu(float x);
__global__ void embedding_kernel(int* token_ids, float* wte, float* wpe, float* embeddings,
                                 int batch_size, int seq_length, int n_embd);
__global__ void qkv_projection_kernel(float* input, float* output,
                                      float* w_qkv, float* b_qkv,
                                      int batch_size, int seq_length, int n_embd);
__global__ void layer_normalization_kernel(float* input,
                                           float* gamma, float* beta,
                                           int batch_size, int seq_length, int n_embd);
__global__ void multi_head_attention_kernel(float* qkv, float* output,
                                            int batch_size, int seq_length, int n_head, int n_embd);
__global__ void final_projection_kernel(float* input, float* output,
                                        float* w_proj, float* b_proj,
                                        int batch_size, int seq_length, int n_embd);
__global__ void add_residual_kernel(float* input, float* residual, float* output,
                                    int batch_size, int seq_length, int n_embd);
__global__ void mlp_kernel(float* input, float* output,
                           float* w_fc, float* b_fc, 
                           float* w_proj, float* b_proj,
                           int batch_size, int seq_length, int n_embd);
__global__ void lm_head_kernel(float* hidden_state, float* logits,
                               float* weights, float* biases,
                               int batch_size, int seq_length, int n_vocab, int n_embd);