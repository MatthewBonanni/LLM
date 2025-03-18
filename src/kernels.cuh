#pragma once

#include <cuda_runtime.h>

#define INTERMEDIATE_SIZE_MAX 4096
#define SEQ_LENGTH_MAX 2048

__device__ __host__ float gelu(float x);
__global__ void embedding_kernel(const int* token_ids,
                                 const float* wte,
                                 const float* wpe,
                                 float* embeddings,
                                 int token_count,
                                 int embedding_dim);
__global__ void qkv_projection_kernel(float* input, float* output,
                                      float* w_qkv, float* b_qkv,
                                      int seq_length, int n_embd, int qkv_size);
__global__ void layer_normalization_kernel(float* input,
                                           float* gamma, float* beta,
                                           int seq_length, int n_embd);
__global__ void multi_head_attention_kernel(float* qkv, float* output,
                                            int seq_length, int n_embd, int n_head);
__global__ void final_projection_kernel(float* input, float* output,
                                        float* w_proj, float* b_proj,
                                        int seq_length, int n_embd);
__global__ void add_residual_kernel(float* input, float* residual, float* output,
                                    int seq_length, int n_embd);
__global__ void mlp_kernel(float* input, float* output, float* w_fc, float* b_fc, 
                           float* w_proj, float* b_proj, int seq_length, int n_embd);
__global__ void lm_head_kernel(float* hidden_state, float* logits,
                               float* weights, float* biases,
                               int n_vocab, int n_embd);