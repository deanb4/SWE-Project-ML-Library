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

Adam::Adam(std::vector<Tensor> params, float lr, float beta1, float beta2, float eps)
    : params(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps) {
        // both running averages start at zero for every element of every param
        for (const Tensor& p : params) {
            m.push_back(std::vector<float>(p.num_elements(), 0.0f));
            v.push_back(std::vector<float>(p.num_elements(), 0.0f));
        }
}

void Adam::step() {
    t++;

    // m and v start at zero, so early on they are too small; dividing by these undoes that
    const float correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    for (size_t i = 0; i < params.size(); i++) {
        Tensor& p = params[i];
        if (!p.get_grad()) {
            continue;
        }
        if (p.get_dtype() != Dtype::FLOAT32) {
            throw std::runtime_error("Adam: only FLOAT32 params are supported");
        }

        // write straight into the params storage, as in SGD
        float* data = static_cast<float*>(p.get_data()->get_ptr());
        const float* grad = static_cast<const float*>(p.get_grad()->get_data()->get_ptr());
        for (size_t j = 0; j < p.num_elements(); j++) {
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * grad[j];
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * grad[j] * grad[j];
            float m_hat = m[i][j] / correction1;
            float v_hat = v[i][j] / correction2;
            data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

void Adam::zero_grad() {
    for (Tensor& p : params) {
        std::shared_ptr<Tensor> none;
        p.set_grad(none);
    }
}