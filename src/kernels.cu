#include "kernels.cuh"

#include <cuda_runtime.h>

__device__ __host__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void embedding_kernel(const int* token_ids,
                                 const float* wte,
                                 const float* wpe,
                                 float* embeddings,
                                 int token_count,
                                 int embedding_dim) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if this thread should process an element
    if (idx < token_count * embedding_dim) {
        // Calculate which token and which embedding dimension this thread is handling
        int token_idx = idx / embedding_dim;    // Which token
        int embd_idx = idx % embedding_dim;     // Which dimension in the embedding

        // Get the token ID for this position
        int token_id = token_ids[token_idx];

        // Calculate offset in embedding tables
        int token_offset = token_id * embedding_dim + embd_idx;
        int pos_offset = token_idx * embedding_dim + embd_idx;

        // Sum token embedding and positional embedding
        embeddings[idx] = wte[token_offset] + wpe[pos_offset];
    }
}

__global__ void qkv_projection_kernel(float* input, float* output,
                                      float* w_qkv, float* b_qkv,
                                      int seq_length, int n_embd, int qkv_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    for (int i = 0; i < qkv_size; i++) {
        float val = b_qkv[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[i_token * n_embd + j] * w_qkv[j * qkv_size + i];
        }
        output[i_token * qkv_size + i] = val;
    }
}

__global__ void layer_normalization_kernel(float* input,
                                           float* gamma, float* beta,
                                           int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        mean += input[i_token * n_embd + i];
    }
    mean /= n_embd;

    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        float diff = input[i_token * n_embd + i] - mean;
        var += diff * diff;
    }
    var /= n_embd;

    // Normalize and scale
    const float epsilon = 1e-5f;
    float inv_std = rsqrtf(var + epsilon);

    for (int i = 0; i < n_embd; i++) {
        float normalized = (input[i_token * n_embd + i] - mean) * inv_std;
        input[i_token * n_embd + i] = gamma[i] * normalized + beta[i];
    }
}

__global__ void multi_head_attention_kernel(float* qkv, float* output,
                                            int seq_length, int n_embd, int n_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
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
            if (j_token <= i_token) {
                // Compute dot product
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    // Q values for token i_token, head i_head
                    // K values for token j_token, head i_head
                    dot += qkv[i_token * qkv_size              + i_head * d_k + d] *
                           qkv[j_token * qkv_size + 1 * n_embd + i_head * d_k + d];
                }
                scores[j_token] = dot * scale;
                max_val = fmaxf(max_val, scores[j_token]);
            }
        }

        // Softmax calculation for attention weights
        float sum = 0.0f;
        for (int j_token = 0; j_token < seq_length; j_token++) {
            // Causal masking: masked tokens have zero weight
            if (j_token <= i_token) {
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
                                qkv[j_token * qkv_size + 2 * n_embd + i_head * d_k + d];
            }
            // Use input as a temporary buffer to store head outputs
            output[i_token * n_embd + i_head * d_k + d] = weighted_sum;
        }
    }
}

__global__ void final_projection_kernel(float* input, float* output,
                                        float* w_proj, float* b_proj,
                                        int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[i_token * n_embd + j] * w_proj[j * n_embd + i];
        }
        output[i_token * n_embd + i] = val;
    }
}

__global__ void add_residual_kernel(float* input, float* residual, float* output, int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }
    
    for (int i = 0; i < n_embd; i++) {
        output[i_token * n_embd + i] = input[i_token * n_embd + i] + residual[i_token * n_embd + i];
    }
}

__global__ void mlp_kernel(float* input, float* output, float* w_fc, float* b_fc, 
                           float* w_proj, float* b_proj, int seq_length, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_token = idx; // Each thread processes one token
    if (i_token >= seq_length) {
        return;
    }

    int intermediate_size = 4 * n_embd;
    float intermediate[INTERMEDIATE_SIZE_MAX];

    // Compute feedforward layer
    for (int i = 0; i < intermediate_size; i++) {
        float val = b_fc[i];
        for (int j = 0; j < n_embd; j++) {
            val += input[i_token * n_embd + j] * w_fc[j * intermediate_size + i];
        }
        intermediate[i] = gelu(val);
    }

    // Compute projection back to hidden size
    for (int i = 0; i < n_embd; i++) {
        float val = b_proj[i];
        for (int j = 0; j < intermediate_size; j++) {
            val += intermediate[j] * w_proj[j * n_embd + i];
        }
        output[i_token * n_embd + i] = val;
    }
}

__global__ void lm_head_kernel(float* hidden_state, float* logits,
                               float* weights, float* biases,
                               int n_vocab, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vocab) {
        logits[idx] = biases ? biases[idx] : 0.0f;
        for (int i = 0; i < n_embd; i++) {
            logits[idx] += hidden_state[i] * weights[idx * n_embd + i];
        }
    }
}