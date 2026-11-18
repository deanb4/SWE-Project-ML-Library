/**
 * @class Sequential
 * @brief Runs layers one after another (similar to torch.nn.Sequential)
 * 
 * Sequential model;
 * model.add(std::make_shared<Linear>(784,128))
 * model.add(std::make_shared<ReLU>())
 * model.add(std::make_shared<Linear>(128,10))
 * 
 * 
 */
#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "mllib/module.h"

class Sequential : public Module {
    private:
        std::vector<std::shared_ptr<Module>> layers;

    public:
        // Adds a layer to the end of the chain and registers its parameters
        void add(std::shared_ptr<Module> layer);

        Tensor forward(const Tensor& x) const override;
};

#endif