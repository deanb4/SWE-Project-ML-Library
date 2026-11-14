#include "mllib/linear.h"
#include "mllib/function.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

    // Random weights from a normal distribution with std = sqrt(2 / in _features)
    Tensor create_weight(size_t in_features, size_t out_features) {
        if (in_features == 0 || out_features == 0) {
            throw std::invalid_argument("Linear: in_features and out features must be greater than 0");
        }

        size_t count = in_features * out_features;
        auto storage = std::make_shared<Storage>(count * sizeof(float), Device::CPU);
        float* values = static_cast<float*>(storage->get_ptr());

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / in_features));
        for (size_t i = 0; i < count; i++) {
            values[i] = distribution(generator);
        }

        std::vector<int64_t> shape = {static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)};
        return Tensor(storage, shape, Device::CPU, true, Dtype::FLOAT32);
    }
}

Linear::Linear(size_t in_features, size_t out_features) : weight(create_weight(in_features, out_features)),
                bias(Tensor::full({1, static_cast<int64_t>(out_features)}, 0.0, Dtype::FLOAT32)) {
    bias.set_requires_grad(true);
}

Tensor Linear::forward(const Tensor& x) const {
    if (x.get_shape().size() != 2) {
        throw std::invalid_argument("Linear: forward input must have shape (N, in_features)");
    }

    Function F("linear", {}, nullptr);

    // add() need equal shapes, so the (1, out) bias is first copied into every row;
    // (N, 1) ones @ (1, out) bias = (N, out)
    int64_t n = x.get_shape()[0];
    Tensor ones = Tensor::full({n, 1}, 1.0, Dtype::FLOAT32);

    return F.add(F.matmul(x, weight), F.matmul(ones, bias));
}

std::vector<Tensor> Linear::parameters() const {
    return {weight, bias};
}