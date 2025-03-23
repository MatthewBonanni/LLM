#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.cuh"

#define WARP_SIZE 32
#define INTERMEDIATE_SIZE 3072
#define SEQ_LENGTH_MAX 1024
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __host__ fp_t gelu(fp_t x);

template <uint32_t BLOCK_SIZE>
__global__ void embedding_kernel(
    const id_t* __restrict__ token_ids,
    const fp_t* __restrict__ wte,
    const fp_t* __restrict__ wpe,
    fp_t* __restrict__ embeddings,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t seq_offset,
    uint32_t n_embd);

template <uint32_t BLOCK_SIZE, uint32_t WARPS_PER_BLOCK>
__global__ void layer_normalization_kernel(
    fp_t* __restrict__ input,
    const fp_t* __restrict__ gamma,
    const fp_t* __restrict__ beta,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void qkv_projection_kernel(
    fp_t* hidden_states,
    fp_t* q,
    half* kv,
    fp_t* w_qkv,
    fp_t* b_qkv,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t seq_offset,
    uint32_t n_embd);

__global__ void multi_head_attention_kernel(
    fp_t* q,
    half* kv,
    fp_t* output,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t seq_offset,
    uint32_t n_head,
    uint32_t n_embd);

__global__ void final_projection_kernel(
    fp_t* input,
    fp_t* output,
    fp_t* w_proj,
    fp_t* b_proj,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void add_residual_kernel(
    fp_t* input,
    fp_t* residual,
    fp_t* output,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void mlp_kernel(
    fp_t* input,
    fp_t* output,
    fp_t* w_fc,
    fp_t* b_fc,
    fp_t* w_proj,
    fp_t* b_proj,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void lm_head_kernel(
    fp_t* hidden_state,
    fp_t* logits,
    fp_t* weights,
    fp_t* biases,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_vocab,
    uint32_t n_embd);
