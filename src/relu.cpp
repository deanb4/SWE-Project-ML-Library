#include "mllib/relu.h"
#include "mllib/function.h"

Tensor ReLU::forward(const Tensor& x) const {
    Function F("relu", {}, nullptr);
    return F.relu(x);
}