#ifndef ATTENTION_H
#define ATTENTION_H

#include <cstddef>
#include "module.h"
#include "linear.h"
#include "tensor.h"

/**
 * @class CasualSelfAttention
 * @brief Multi-head casual self-attention: the part of a GPT block wehre each token
 * gathers semantic/contextual information from the tockens around it.
 * 
 * input x: (B, T, C) B sequences, T tokens each, C = embed_dim numbers per token
 * output: (B, T, C) same shape, so blocks stack and the result can be added to x
 * 
 * Each head works on head_size / C / num_heads of the C columns:
 * q = x @ Wq what each token is looking for
 * k = x @ Wk what each token offers to 
 * v = X @ Wv what each token passes on if picked
 * scores = q @ k^T / sqrt(head_size) (B, T, T): how well token i's query matches token j's key
 * scores[i][j] = -infinity for j > i casual mask: a token can't see past where it is (can't cheat by looking ahead)
 * weights = softmax over j of scores : each row sums to 1
 * out = weights @ v : (B, T, head_size): a weighted mix of earlier tokens
 * the heads outputs site side by side (C columns again) and proj mixes them.
 * 
 * All heads are computed at once: query/key/values are (C,C) Linaer layers whose columns hold
 * every head's weights side by side, then a reshape splits the columns into heads. 
 */

class CausalSelfAttention : public Module {
    private:
        Linear query; // (C,C) : every head's Wq side by side
        Linear Key; // (C, C)
        Linear value;
        Linear proj; // (C,C): mixes the heads outputs back together
        size_t num_heads;
        size_t head_size; // C / num_heads
    public:
    /**
     * @throws std::invalid_argument if embed_dim or num_heads is zero, or embed_dim is not 
     * divisible by num_heads
     */
    CausalSelfAttention(size_t embed_dim, size_t num_heads);

    /**
     * @param x FLOAT32 tensor of shape (B,T,C)
     * @return Tensor of shape (B,T,C)
     */
    Tensor forward(const Tensor& x) const override;
};

#endif