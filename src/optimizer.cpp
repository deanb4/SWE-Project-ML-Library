#include "mllib/optimizer.h"
#include <stdexcept>

SGD::SGD(std::vector<Tensor> params, float lr) : params(params), lr(lr) {}

void SGD::step() {
    for (Tensor& p : params) {
        if (!p.get_grad()) {
            continue;
        }
        if (p.get_dtype() != Dtype::FLOAT32) {
            throw std::runtime_error("SGD: only FLOAT32 paramaters are supported");
        }

        // Write straight into the paramter's storage. Using ops (sub, mul) would create a 
        // new tensor, and the layer would keep using its old weights.
        float* data = static_cast<float*>(p.get_data()->get_ptr());
        const float* grad = static_cast<const float*>(p.get_grad()->get_data()->get_ptr());
        for (size_t i = 0; i < p.num_elements(); ++i) {
            data[i] -= lr * grad[i];
        }
    }
}

void SGD::zero_grad() {
    for (Tensor& p : params) {
        std::shared_ptr<Tensor> none;
        p.set_grad(none);
    }
}