// autograd engine
#include "mllib/autograd.h"
#include "mllib/function.h"
#include <algorithm>
#include <unordered_set>
#include <stdexcept>

bool Autograd::grad_enabled = true;

namespace {

    // Depth first walk: a tensor is added only after every tensor is computer from.
    // Leaves (no grad_fn) are skipped; backward writes their grads without visiting them.
    // Every copy of an op's output points at the same Function, so the Function 
    // identifies the tensor. 

    void visit(const std::shared_ptr<Tensor>& t, std::unordered_set<Function*>& visited,
                std::vector<std::shared_ptr<Tensor>>& order) {

        Function* fn = t->get_grad_fn().get();
        if (fn == nullptr || visited.count(fn) > 0) {
            return;
        }
        visited.insert(fn);

        for (auto input : fn->get_inputs()) {
            visit(input, visited, order);
        }
        order.push_back(t);
    }

    // t.grad += g( or t.grad = g if t has no grad yet)
    void accumulate_grad(Tensor& t, const Tensor& g) {
        std::shared_ptr<Tensor> total;
        if (t.get_grad()) {
            Function F("accumulate", {}, nullptr);
            total = std::make_shared<Tensor>(F.add(*t.get_grad(), g));

        } else {
            total = std::make_shared<Tensor>(g);
        }
        t.set_grad(total);
    }

    // Turns grad trracking off until it goes out of scope (also if backwards throws error)
    struct NoGrad {
        bool was_enabled = Autograd::is_grad_enabled();
        NoGrad() { Autograd::set_grad_enabled(false); }
        ~NoGrad() { Autograd::set_grad_enabled(was_enabled); }
    };

}

bool Autograd::is_grad_enabled() {
    return grad_enabled;
}

void Autograd::set_grad_enabled(bool value) {
    grad_enabled = value;
}

std::vector<std::shared_ptr<Tensor>> Autograd::topological_sort(std::shared_ptr<Tensor> root) {
    std::vector<std::shared_ptr<Tensor>> order;
    std::unordered_set<Function*> visited;
    visit(root, visited, order);

    // visit() puts inputs before outputs; backward needs the root first
    std::reverse(order.begin(), order.end());
    return order;
} 


void Autograd::backward(std::shared_ptr<Tensor> root) {
    // the gradient math below uses the same ops
    NoGrad no_grad;

    std::vector<std::shared_ptr<Tensor>> order = topological_sort(root);
    
    for (std::shared_ptr<Tensor>& t : order) {
        std::shared_ptr<Tensor> none;
        t->set_grad(none);
    }

    accumulate_grad(*root, Tensor::full(root->get_shape(), 1.0, root->get_dtype()));

    for (std::shared_ptr<Tensor>& t : order) {
        if (!t->get_grad()) {
            continue; 
        }

        std::shared_ptr<Function> fn = t->get_grad_fn();
        std::vector<Tensor> input_grads = fn->backward(*t->get_grad());

        const std::vector<std::shared_ptr<Tensor>>& inputs = fn->get_inputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i]->get_requires_grad()) {
                accumulate_grad(*inputs[i], input_grads[i]);
            }
        }
    }
}