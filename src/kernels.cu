#include "kernels.cuh"

#include <cuda_runtime.h>

#include "utils.cuh"

__device__ __host__ fp_t gelu(fp_t x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void embedding_kernel(
        id_t* token_ids,
        fp_t* wte,
        fp_t* wpe,
        fp_t* embeddings,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }
    
    // Get token ID for current position
    id_t token_id = token_ids[idx_batch * seq_length + idx_seq];

    // Calculate offsets
    uint64_t offset_out = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;
    uint64_t offset_wte = (uint64_t)token_id * n_embd;
    uint64_t offset_wpe = (uint64_t)idx_seq * n_embd;

    // Perform embedding lookup
    for (uint32_t i = 0; i < n_embd; i++) {
        embeddings[offset_out + i] = wte[offset_wte + i] + wpe[offset_wpe + i];
    }
}

__global__ void layer_normalization_kernel(
        fp_t* input,
        fp_t* gamma,
        fp_t* beta,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    uint64_t offset_input = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

    // Calculate mean
    fp_t mean = 0.0f;
    for (uint32_t i = 0; i < n_embd; i++) {
        mean += input[offset_input + i];
    }
    mean /= n_embd;

    // Calculate variance
    fp_t var = 0.0f;
    for (uint32_t i = 0; i < n_embd; i++) {
        fp_t diff = input[offset_input + i] - mean;
        var += diff * diff;
    }
    var /= n_embd;

    // Normalize and scale
    const fp_t epsilon = 1e-5f;
    fp_t inv_std = rsqrtf(var + epsilon);

    for (uint32_t i = 0; i < n_embd; i++) {
        fp_t normalized = (input[offset_input + i] - mean) * inv_std;
        input[offset_input + i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void qkv_projection_kernel(
        fp_t* input,
        fp_t* output,
        fp_t* w_qkv,
        fp_t* b_qkv,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    uint64_t offset_input  = ((uint64_t) idx_batch * seq_length + idx_seq) * n_embd;
    uint64_t offset_output = ((uint64_t) idx_batch * seq_length + idx_seq) * (3 * n_embd);
    uint32_t qkv_size = 3 * n_embd; // Size of Q, K, V for each token

    // Perform QKV projection
    for (uint32_t i = 0; i < qkv_size; i++) {
        fp_t val = b_qkv[i];
        for (uint32_t j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_qkv[j * qkv_size + i];
        }
        output[offset_output + i] = val;
    }
}

__global__ void multi_head_attention_kernel(
        fp_t* qkv,
        fp_t* output,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_head,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Calculate dimensions
    uint32_t d_k = n_embd / n_head; // Dimension of each head
    uint32_t qkv_size = 3 * n_embd; // Size of Q, K, V for each token

    // Scores register
    fp_t scores[SEQ_LENGTH_MAX];

    fp_t scale = rsqrtf(d_k);

    // Process each attention head
    for (uint32_t i_head = 0; i_head < n_head; i_head++) {
        // Calculate attention scores between current token and all other tokens
        fp_t max_val = -INFINITY;
        for (uint32_t j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: only attend to positions j_token <= i_token
            if (j_token <= idx_seq) {
                // Compute dot product
                fp_t dot = 0.0f;
                for (uint32_t d = 0; d < d_k; d++) {
                    // Q values for token idx_seq, head i_head
                    // K values for token j_token, head i_head
                    dot += qkv[idx_batch * seq_length * qkv_size + idx_seq * qkv_size + i_head * d_k + d] *
                           qkv[idx_batch * seq_length * qkv_size + j_token * qkv_size + 1 * n_embd + i_head * d_k + d];
                }
                scores[j_token] = dot * scale;
                max_val = fmaxf(max_val, scores[j_token]);
            }
        }

        // Softmax calculation for attention weights
        fp_t sum = 0.0f;
        for (uint32_t j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: masked tokens have zero weight
            if (j_token <= idx_seq) {
                scores[j_token] = expf(scores[j_token] - max_val);
                sum += scores[j_token];
            } else {
                scores[j_token] = 0.0f;
            }
        }

        for (uint32_t j_token = 0; j_token < seq_length; j_token++) {
            scores[j_token] /= sum;
        }

        // Calculate weighted sum of values
        for (uint32_t d = 0; d < d_k; d++) {
            fp_t weighted_sum = 0.0f;
            for (uint32_t j_token = 0; j_token < seq_length; j_token++) {
                // Get V values for token j_token, head i_head
                weighted_sum += scores[j_token] *
                                qkv[idx_batch * seq_length * qkv_size + j_token * qkv_size + 2 * n_embd + i_head * d_k + d];
            }
            // Use input as a temporary buffer to store head outputs
            output[idx_batch * seq_length * n_embd + idx_seq * n_embd + i_head * d_k + d] = weighted_sum;
        }
    }
}

__global__ void final_projection_kernel(
        fp_t* input,
        fp_t* output,
        fp_t* w_proj,
        fp_t* b_proj,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    uint64_t offset_input = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

    // Perform final projection
    for (uint32_t i = 0; i < n_embd; i++) {
        fp_t val = b_proj[i];
        for (uint32_t j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_proj[j * n_embd + i];
        }
        output[offset_input + i] = val;
    }
}

__global__ void add_residual_kernel(
        fp_t* input,
        fp_t* residual,
        fp_t* output,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    uint64_t offset = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

    // Add residual connection
    for (uint32_t i = 0; i < n_embd; i++) {
        output[offset + i] = input[offset + i] + residual[offset + i];
    }
}

__global__ void mlp_kernel(
        fp_t* input,
        fp_t* output,
        fp_t* w_fc,
        fp_t* b_fc, 
        fp_t* w_proj,
        fp_t* b_proj,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Intermediate register
    uint32_t intermediate_size = 4 * n_embd;
    fp_t intermediate[INTERMEDIATE_SIZE_MAX];

    // Get the starting index for the current token
    uint64_t offset_input = ((uint64_t)idx_batch * seq_length + idx_seq) * n_embd;

    // Compute feedforward layer
    for (uint32_t i = 0; i < intermediate_size; i++) {
        fp_t val = b_fc[i];
        for (uint32_t j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_fc[j * intermediate_size + i];
        }
        intermediate[i] = gelu(val);
    }

    // Compute projection back to hidden size
    for (uint32_t i = 0; i < n_embd; i++) {
        fp_t val = b_proj[i];
        for (uint32_t j = 0; j < intermediate_size; j++) {
            val += intermediate[j] * w_proj[j * n_embd + i];
        }
        output[offset_input + i] = val;
    }
}

__global__ void lm_head_kernel(
        fp_t* hidden_state,
        fp_t* logits,
        fp_t* weights,
        fp_t* biases,
        uint32_t batch_size,
        uint32_t seq_length,
        uint32_t n_vocab,
        uint32_t n_embd) {
    // Calculate thread ID
    uint32_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx_vocab = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_vocab >= n_vocab) {
        return;
    }

    // Calculate output index
    uint64_t idx_out = ((uint64_t)idx_batch * n_vocab + idx_vocab);

    // Get the starting index for the current token
    uint64_t offset_input =  ((uint64_t)idx_batch * seq_length + (seq_length - 1)) * n_embd;
    uint64_t offset_weights = (uint64_t)idx_vocab * n_embd;

    // Compute logits
    logits[idx_out] = biases ? biases[idx_vocab] : 0.0f;
    for (uint32_t i = 0; i < n_embd; i++) {
        logits[idx_out] += hidden_state[offset_input + i] * weights[offset_weights + i];
    }
}