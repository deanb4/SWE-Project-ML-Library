/**
 * @class ReLU
 * @brief relu as a layer, so it can go in sequential. (has no params)
 * 
 */

 #ifndef RELU_H
 #define RELU_H

#include "mllib/module.h"

 class ReLU: public Module {
    public:
        Tensor forward(const Tensor& x) const override;
 };

 #endif