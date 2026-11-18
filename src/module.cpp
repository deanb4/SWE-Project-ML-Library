#include "mllib/module.h"

void Module::register_module(const Module& child) {
    // Tensor copies share storage and grads with the layer's own tensors,
    // so the optimizier still updates the layers weights through these copies
    for (const Tensor& p : child.parameters()) {
        params.push_back(p);
    }
}

Tensor Module::operator()(const Tensor& x) const {
    return forward(x);
}

std::vector<Tensor> Module::parameters() const {
    return params;
}