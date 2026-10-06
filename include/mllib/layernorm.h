#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <vector>
#include "tensor.h"

/**
 * @class LayerNorm
 * @brief Applies layer normalization to the last dimension of a tensor.
 *
 * LayerNorm normalizes each token's feature vector so that it has zero mean and unit variance,
 * then applies a learned scale and shift.
 *
 * For a token vector x with D features, the mean is computed as:
 *
 *     mean = (1 / D) * sum of all x[j]
 *
 * The variance is computed as:
 *
 *     variance = (1 / D) * sum of all (x[j] - mean)^2
 *
 * The normalized value for each feature is:
 *
 *     x_hat[j] = (x[j] - mean) / sqrt(variance + eps)
 *
 * The final output is:
 *
 *     y[j] = gamma[j] * x_hat[j] + beta[j]
 *
 * Normalization is performed over the last dimension only. For example, an input with shape
 * (T, D) is normalized independently for each of its T token vectors, while an input with 
 * shape (B, T, D) is normalized independently for each (B, T) token.
 *
 * The layer has two learnable parameters: gamma (learned scale) and beta (learned shift).
 *
 * Gamma is initialized to 1 and beta is initialized to 0. This means that before training, 
 * the layer performs plain normalization without changing the normalized values.
 *
 * The small constant eps is added to the variance before taking the square root to prevent 
 * division by zero when all values in a token vector are identical.
 *
 * The backward pass does not require a custom implementation. The forward computation can be 
 * constructed from existing tensor operations such as mean, sub, mul, sqrt, div, and add, 
 * allowing the autograd system to compute gradients automatically.
 */
class LayerNorm {
    private:

        /**
         * @brief Learned scale parameter.
         *
         * Each feature has its own multiplicative scale. The tensor has
         * shape (dim) and is initialized to 1.
         *
         * Gamma is shared across all tokens being normalized.
         */
        Tensor gamma;

        /**
         * @brief Learned shift parameter.
         *
         * Each feature has its own additive bias. The tensor has shape
         * (dim) and is initialized to 0.
         *
         * Beta is shared across all tokens being normalized.
         */
        Tensor beta;

        /**
         * @brief Small constant added to the variance for numerical stability.
         *
         * The epsilon value prevents division by zero when the variance is
         * zero or extremely close to zero.
         */
        double eps;

    public:

        /**
         * @brief Constructs a LayerNorm module.
         *
         * Creates the learnable gamma and beta parameters, each with shape (dim).
         *
         * Gamma is initialized to 1 so that it initially leaves the scale of normalized values
         * unchanged. Beta is initialized to 0 so that it initially introduces no shift.
         *
         * @param dim Number of features in the dimension being normalized.
         * @param eps Small constant added to the variance to prevent division by zero. 
         *            Defeault value is 1e-5.
         *
         * @throws std::invalid_argument If dim is zero or otherwise invalid.
         */
        LayerNorm(size_t dim, double eps = 0.00001);

        /**
         * @brief Applies layer normalization to an input tensor.
         *
         * The input is normalized over its last dimension only. For an input whose final 
         * dimension has size dim, each vector along that dimension is normalized independently.
         *
         * For each vector x:
         *
         *     mean = mean(x)
         *
         *     variance = mean((x - mean)^2)
         *
         *     x_hat = (x - mean) / sqrt(variance + eps)
         *
         *     y = gamma * x_hat + beta
         *
         * Gamma and beta are broadcast across all preceding dimensions.
         *
         * For example:
         *
         *     input shape:  (T, D)
         *     output shape: (T, D)
         *
         * or:
         *
         *     input shape:  (B, T, D)
         *     output shape: (B, T, D)
         *
         * @param x Input tensor whose last dimension has size dim.
         * @return A tensor with the same shape as x, containing the normalized and 
         *         affine-transformed values.
         */
        Tensor forward(const Tensor& x) const;

        /**
         * @brief Returns the learnable parameters of this LayerNorm.
         *
         * LayerNorm has two learnable parameters: gamma (learned scale) and beta (learned shift).
         *
         * @return A vector containing gamma and beta.
         */
        std::vector<Tensor> parameters() const;
};

#endif