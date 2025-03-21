#include "kernels.cuh"

#include <cuda_runtime.h>

__device__ __host__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void embedding_kernel(int* token_ids, float* wte, float* wpe, float* embeddings,
                                 int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }
    
    // Get token ID for current position
    int token_id = token_ids[idx_batch * seq_length + idx_seq];

    // Calculate offsets
    size_t offset_out = (idx_batch * seq_length + idx_seq) * n_embd;
    size_t offset_wte = token_id * n_embd;
    size_t offset_wpe = idx_seq * n_embd;

    // Perform embedding lookup
    for (int i = 0; i < n_embd; i++) {
        embeddings[offset_out + i] = wte[offset_wte + i] + wpe[offset_wpe + i];
    }
}

__global__ void qkv_projection_kernel(float* input, float* output,
                                      float* w_qkv, float* b_qkv,
                                      int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    size_t offset_input = (idx_batch * seq_length + idx_seq) * n_embd;
    size_t offset_output = (idx_batch * seq_length + idx_seq) * (3 * n_embd);
    int qkv_size = 3 * n_embd; // Size of Q, K, V for each token

    // Perform QKV projection
    for (int i = 0; i < qkv_size; i++) {
        float val = b_qkv[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_qkv[j * qkv_size + i];
        }
        output[offset_output + i] = val;
    }
}

__global__ void layer_normalization_kernel(float* input,
                                           float* gamma, float* beta,
                                           int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    size_t offset_input = (idx_batch * seq_length + idx_seq) * n_embd;

    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        mean += input[offset_input + i];
    }
    mean /= n_embd;

    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        float diff = input[offset_input + i] - mean;
        var += diff * diff;
    }
    var /= n_embd;

    // Normalize and scale
    const float epsilon = 1e-5f;
    float inv_std = rsqrtf(var + epsilon);

    for (int i = 0; i < n_embd; i++) {
        float normalized = (input[offset_input + i] - mean) * inv_std;
        input[offset_input + i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void multi_head_attention_kernel(float* qkv, float* output,
                                            int batch_size, int seq_length, int n_head, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Calculate dimensions
    int d_k = n_embd / n_head; // Dimension of each head
    int qkv_size = 3 * n_embd; // Size of Q, K, V for each token

    // Scores register
    float scores[SEQ_LENGTH_MAX];

    float scale = rsqrtf(d_k);

    // Process each attention head
    for (int i_head = 0; i_head < n_head; i_head++) {
        // Calculate attention scores between current token and all other tokens
        float max_val = -INFINITY;
        for (int j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: only attend to positions j_token <= i_token
            if (j_token <= idx_seq) {
                // Compute dot product
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
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
        float sum = 0.0f;
        for (int j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: masked tokens have zero weight
            if (j_token <= idx_seq) {
                scores[j_token] = expf(scores[j_token] - max_val);
                sum += scores[j_token];
            } else {
                scores[j_token] = 0.0f;
            }
        }

        for (int j_token = 0; j_token < seq_length; j_token++) {
            scores[j_token] /= sum;
        }

        // Calculate weighted sum of values
        for (int d = 0; d < d_k; d++) {
            float weighted_sum = 0.0f;
            for (int j_token = 0; j_token < seq_length; j_token++) {
                // Get V values for token j_token, head i_head
                weighted_sum += scores[j_token] *
                                qkv[idx_batch * seq_length * qkv_size + j_token * qkv_size + 2 * n_embd + i_head * d_k + d];
            }
            // Use input as a temporary buffer to store head outputs
            output[idx_batch * seq_length * n_embd + idx_seq * n_embd + i_head * d_k + d] = weighted_sum;
        }
    }
}

__global__ void final_projection_kernel(float* input, float* output,
                                        float* w_proj, float* b_proj,
                                        int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    size_t offset_input = (idx_batch * seq_length + idx_seq) * n_embd;

    // Perform final projection
    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_proj[j * n_embd + i];
        }
        output[offset_input + i] = val;
    }
}

__global__ void add_residual_kernel(float* input, float* residual, float* output,
                                    int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Get the starting index for the current token
    size_t offset = (idx_batch * seq_length + idx_seq) * n_embd;

    // Add residual connection
    for (int i = 0; i < n_embd; i++) {
        output[offset + i] = input[offset + i] + residual[offset + i];
    }
}

__global__ void mlp_kernel(float* input, float* output,
                           float* w_fc, float* b_fc, 
                           float* w_proj, float* b_proj,
                           int batch_size, int seq_length, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_seq   = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_seq   >= seq_length) {
        return;
    }

    // Intermediate register
    int intermediate_size = 4 * n_embd;
    float intermediate[INTERMEDIATE_SIZE_MAX];

    // Get the starting index for the current token
    size_t offset_input = (idx_batch * seq_length + idx_seq) * n_embd;

    // Compute feedforward layer
    for (int i = 0; i < intermediate_size; i++) {
        float val = b_fc[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[offset_input + j] * w_fc[j * intermediate_size + i];
        }
        intermediate[i] = gelu(val);
    }

    // Compute projection back to hidden size
    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int j = 0; j < intermediate_size; j++) {
            val += intermediate[j] * w_proj[j * n_embd + i];
        }
        output[offset_input + i] = val;
    }
}

__global__ void lm_head_kernel(float* hidden_state, float* logits,
                               float* weights, float* biases,
                               int batch_size, int seq_length, int n_vocab, int n_embd) {
    // Calculate thread ID
    size_t idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_vocab = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (idx_batch >= batch_size ||
        idx_vocab >= n_vocab) {
        return;
    }

    // Calculate output index
    size_t idx_out = (idx_batch * n_vocab + idx_vocab);

    // Get the starting index for the current token
    size_t offset_input =  (idx_batch * seq_length + (seq_length - 1)) * n_embd;
    size_t offset_weights = idx_vocab * n_embd;

    // Compute logits
    logits[idx_out] = biases ? biases[idx_vocab] : 0.0f;
    for (int i = 0; i < n_embd; i++) {
        logits[idx_out] += hidden_state[offset_input + i] * weights[offset_weights + i];
    }
}