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

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
__global__ void q_projection_kernel(
    const fp_t* __restrict__ hidden_states,
    fp_t* __restrict__ q,
    const fp_t* __restrict__ w_q,
    const fp_t* __restrict__ b_q,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
__global__ void kv_projection_kernel(
    const fp_t* __restrict__ hidden_states,
    half* __restrict__ kv,
    const fp_t* __restrict__ w_kv,
    const fp_t* __restrict__ b_kv,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t seq_offset,
    uint32_t n_embd);

__global__ void multi_head_attention_kernel(
    const fp_t* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    fp_t* __restrict__ output,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t seq_offset,
    uint32_t n_head,
    uint32_t n_embd);

__global__ void final_projection_kernel(
    const fp_t* __restrict__ input,
    fp_t* __restrict__ output,
    const fp_t* __restrict__ w_proj,
    const fp_t* __restrict__ b_proj,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

template <uint32_t BLOCK_SIZE>
__global__ void add_residual_kernel(
    const fp_t* __restrict__ input,
    const fp_t* __restrict__ residual,
    fp_t* __restrict__ output,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void mlp_kernel(
    const fp_t* __restrict__ input,
    fp_t* __restrict__ output,
    const fp_t* __restrict__ w_fc,
    const fp_t* __restrict__ b_fc,
    const fp_t* __restrict__ w_proj,
    const fp_t* __restrict__ b_proj,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_embd);

__global__ void lm_head_kernel(
    const fp_t* __restrict__ hidden_state,
    fp_t* __restrict__ logits,
    const fp_t* __restrict__ weights,
    const fp_t* __restrict__ biases,
    uint32_t batch_size,
    uint32_t seq_length,
    uint32_t n_vocab,
    uint32_t n_embd);
