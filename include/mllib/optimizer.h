#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>

/**
 * @class SGD
 * @brief plain stochastic gradient descent: param = param - lr * param.grad
 * 
 * The paramaters are copies of the layers' tensors. Copies share storage and grads, so
 * step() changes the layers' own weights.
 * 
 * Training loop:
 * 
 *  optimizer.zero_grad();
 *  loss.backward();
 *  optimizer.step();
 * 
 */

class SGD {
    private:
        std::vector<Tensor> params;
        float lr;

    public:
        SGD(std::vector<Tensor> params, float lr);

        // Moves each parameter a step against its gradient. Parameters without a grad are skipped.
        void step();

        // Clears every grad, so the next backward() starts from zero instead of adding on
        void zero_grad();
};

#endif