#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <vector>
#include <cstddef>
#include "tensor.h"

/**
 * @class Embedding
 * @brief A learnable embedding table that maps token IDs to vectors.
 * 
 * The Embedding class stores a table of learned vectors, where each row corresponds to a 
 * token in the vocabulary.
 * 
 * If the vocabulary size is `vocab_size` and each embedding has dimension `embed_dim`, the
 * embedding table has the shape of (vocab_size, embed_dim).
 * 
 * Given a tensor of integer token IDs, the forward pass looks up the corresponding row of
 * the embedding table for each ID.
 * 
 * For example, if the input is: [2, 5, 2]
 * then the output consists of: [weight[2], weight[5], weight[2]]
 * 
 * Token IDs are indices rather than numerical values, so they are not themselves learnable
 * and do not receive gradients. During backpropagation, gradients from repeated token IDs 
 * are accumulated into the corresponding row of the embedding table.
 */
class Embedding {
    private:

        /**
         * @brief Learnable embedding table.
         * 
         * Each row represents the embedding vector associated with one token ID. The tensor
         * has the shape of (vocab_size, embed_dim). 
         */
        Tensor weight;

    public:

        /**
         * @brief Constructs an embedding table.
         * 
         * Creates a learnable embedding table with `vocab_size` rows and `embed_dim` columns.
         * The table should be initialized with random values rather than zeros. 
         * 
         * @param vocab_size The number of unique tokens in the vocabulary. 
         * @param embed_dim The dimensionality of each token's embedding vector.
         */
        Embedding(size_t vocab_size, size_t embed_dim);

        /**
         * @brief Looks up the embedding vector for each token ID.
         * 
         * For an input tensor containing token IDs, this function returns the corresponding 
         * rows from the embedding table.
         * 
         * If `ids` has shape `(T)`, the output has shape: (T, embed_dim)
         * If `ids` has shape `(B, T)`, the output has shape: (B, T, embed_dim)
         * 
         * Each token ID must be in the range `[0, vocab_size)`. An invalid ID should result in 
         * an error rather than an out-of-bounds access.
         * 
         * @param ids A tensor containing integer token IDs.
         * @return A tensor containing the corresponding embedding vectors.
         */
        Tensor forward(const Tensor& ids) const;

        /**
         * @brief Returns the learnable parameters of this embedding.
         * 
         * The embedding module has a single learnable parameter: the embedding table stored in `weight`.
         * 
         * @return A shared pointer to the embedding table.
         */
        std::shared_ptr<Tensor> parameters() const;
};


#endif