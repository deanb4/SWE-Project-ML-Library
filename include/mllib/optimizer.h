#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <cmath>

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

/**
 * @class Adam
 * @brief SGD with two running averages per weight 
 * 
 * m = beta1 * m + (1 - beta1) * grad (average gradient; which way to go)
 * v = beta2 * v (1 - beta2) * grad^2 (how big grads are)
 * param -= lr * m_hat / (sqrt(v_hat) + eps)
 * 
 * Dividing by sqrt(v) gives every weight its own step size, so weights with tiny grads
 * still move
 * 
 * Unlike SGD, Ada, keeps m and v between steps, so has to be created once prior 
 * to training not per epoch!
 */

 class Adam {
    private:
        std::vector<Tensor> params;
        float lr;
        float beta1;
        float beta2;
        float eps;
        int64_t t = 0;
        std::vector<std::vector<float>> m;
        std::vector<std::vector<float>> v;

    public:
        Adam(std::vector<Tensor> params, float lr = 0.001f, float beta1 = 0.9f,
            float beta2 = 0.999f, float eps = 1e-8f);

        // Moves each param using its grad and running avgs. params w/out grad skipped
        void step();

        // clears every grad;
        void zero_grad();
};

#endif