#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

#define INTERMEDIATE_SIZE_MAX 3072
#define SEQ_LENGTH_MAX 1024

__device__ __host__ fp_t gelu(fp_t x);

__global__ void embedding_kernel(
    id_t* token_ids,
    fp_t* wte,
    fp_t* wpe,
    fp_t* embeddings,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void qkv_projection_kernel(
    fp_t* input,
    fp_t* output,
    fp_t* w_qkv,
    fp_t* b_qkv,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void layer_normalization_kernel(
    fp_t* input,
    fp_t* gamma,
    fp_t* beta,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void multi_head_attention_kernel(
    fp_t* qkv,
    fp_t* output,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_head,
    uint64_t n_embd);

__global__ void final_projection_kernel(
    fp_t* input,
    fp_t* output,
    fp_t* w_proj,
    fp_t* b_proj,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void add_residual_kernel(
    fp_t* input,
    fp_t* residual,
    fp_t* output,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void mlp_kernel(
    fp_t* input,
    fp_t* output,
    fp_t* w_fc,
    fp_t* b_fc,
    fp_t* w_proj,
    fp_t* b_proj,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_embd);

__global__ void lm_head_kernel(
    fp_t* hidden_state,
    fp_t* logits,
    fp_t* weights,
    fp_t* biases,
    uint64_t batch_size,
    uint64_t seq_length,
    uint64_t n_vocab,
    uint64_t n_embd);