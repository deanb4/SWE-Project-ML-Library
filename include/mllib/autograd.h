// Backwards propagation 
#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <vector>
#include "tensor.h"
#include <memory>

class Autograd {
    private:
        static bool grad_enabled;
        static std::vector<std::shared_ptr<Tensor>> topological_sort(std::shared_ptr<Tensor> root);

    public:
        static bool is_grad_enabled();
        static void set_grad_enabled(bool value); 
        static void backward(std::shared_ptr<Tensor> root);
};

#endif