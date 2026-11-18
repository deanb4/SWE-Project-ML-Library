#ifndef MODULE_H
#define MODULE_H

#include <memory>
#include <vector>
#include "tensor.h"

/**
 * @class Module
 * @brief Base class for layers and models
 * 
 * struct Net : Module {
 * Lienar fc1{784, 128}}
 * Lienar fc2{128, 10}}
 * Net() {
 *  register_module(fc1);
 *  register_module(fc2);
 * }
 * Tensor forward(const Tensor& x) const override {
 * Function F("net", {}, nullptr)};
 * return fc2(F.relu(fc1(x)));
 * 
 * A layer that is not registered still runs in forward(), but its weights are missing
 * from parameters(), so the optimizer never updates them
 */

 class Module {

    private:
        std::vector<Tensor> params;
    protected:
        // adds every paramter of a layer to this module's list.
        // Call in constructor once for each layer
        void register_module(const Module& child);

    public:
        virtual ~Module() = default;
        
        // runes the layer or model on x
        virtual Tensor forward(const Tensor& x) const = 0;

        // Turns model.forward(x) into model(x) 
        Tensor operator()(const Tensor& x) const;

        // Every learnable tensor, ready to pass to an optimizer. Layers that own their
        // weights (Linear, Conv2d) override this to return them; models reutrn what they registered.
        virtual std::vector<Tensor> parameters() const;

 };

 
 #endif