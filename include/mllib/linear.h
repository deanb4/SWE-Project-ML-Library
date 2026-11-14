#ifndef LINEAR_H
#define LINEAR_H

#include <cstddef>
#include <vector>
#include "tensor.h"

/** 
 * @class linear
 * @brief a fully connected layer: out = x @ weight + bias (@ = matric multiplication)
 * 
 * input x has shape (N, in_features), one row per example, and the output has shape
 * (N, out_features).
 * 
 * The weight is initialized with Kaiming (He) initialization, std = sqrt( 2 / in_features),
 * which suits layers followed by rely. The bias starts at zero.
 */

class Linear {
    private:
        Tensor weight;
        Tensor bias;

    public:
    /**
     * @thorws std::invalid_argument if in_features or out_features is zero.
     */
    Linear(size_t in_faetures, size_t out_features);

    /**
     * @param x FLOAT32 tensor of shape (N, in_features)
     * @return Tensor of shape (N, out_features)
     */
    Tensor forward(const Tensor& x) const;

    /**
     * @return {weight, biasd}. They share storage and grads with the layer's own tensors, 
     * so an optimizer can update them
     */
    std::vector<Tensor> parameters() const;
};

#endif