#include <mllib/sequential.h>

void Sequential::add(std::shared_ptr<Module> layer) {
    register_module(*layer);
    layers.push_back(layer);
}

Tensor Sequential::forward(const Tensor& x) const {
    Tensor out = x;
    for (const std::shared_ptr<Module>& layer : layers) {
        out = layer->forward(out);
    }
    return out;
}