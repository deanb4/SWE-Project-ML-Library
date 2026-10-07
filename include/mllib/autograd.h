#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <vector>
#include "tensor.h"
#include <memory>

/**
 * @class Autograd
 * @brief Global engine that runs backpropagation over the computation graph.
 *
 * When an operation is applied to tensors that require gradients, the resulting tensor
 * records the operation that produced it in its `grad_fn`. Following `grad_fn` links from
 * an output tensor back to its inputs forms a directed acyclic graph of the computation.
 *
 * Autograd walks this graph in reverse to compute the gradient of the root tensor with
 * respect to every tensor in the graph that requires gradients.
 *
 * For example, if: y = a * b and loss = y + c
 * then calling backward on `loss` computes: d(loss)/da, d(loss)/db, and d(loss)/dc
 *
 * All members are static, so Autograd is never instantiated. Gradient tracking can be
 * turned off globally (for example during inference) to avoid building the graph.
 */
class Autograd {
    private:

        /**
         * @brief Whether gradient tracking is currently enabled.
         *
         * When true, operations on tensors that require gradients record their `grad_fn`
         * so the graph can be traversed during backpropagation. When false, no graph is
         * built. Enabled by default.
         */
        static bool grad_enabled;

        /**
         * @brief Orders the tensors in the computation graph for backpropagation.
         *
         * Performs a depth-first traversal starting at `root` and following each tensor's
         * `grad_fn` to its inputs. Each tensor appears in the result exactly once, even if it
         * is used by several operations.
         *
         * The result is ordered so that every tensor comes before the tensors it was computed
         * from. This guarantees a tensor's gradient is fully accumulated from all of its uses
         * before it is propagated further back to its inputs.
         *
         * @param root The tensor to start the traversal from, typically the loss.
         * @return The tensors in the graph, ordered from `root` back to the leaf tensors.
         */
        static std::vector<std::shared_ptr<Tensor>> topological_sort(std::shared_ptr<Tensor> root);

    public:

        /**
         * @brief Returns whether gradient tracking is currently enabled.
         *
         * @return True if operations record their `grad_fn`, false otherwise.
         */
        static bool is_grad_enabled();

        /**
         * @brief Enables or disables gradient tracking globally.
         *
         * Disabling gradient tracking stops operations from building the computation graph,
         * which saves memory and time when gradients are not needed, such as during inference
         * or when updating parameters in an optimizer step.
         *
         * @param value True to enable gradient tracking, false to disable it.
         */
        static void set_grad_enabled(bool value);

        /**
         * @brief Runs backpropagation from the given root tensor.
         *
         * Sets the gradient of `root` to ones with the same shape as `root`, then visits the
         * tensors in the order returned by `topological_sort`. For each tensor, its `grad_fn`
         * computes the gradients of its inputs, which are accumulated (added) into each
         * input's `grad`.
         *
         * Only tensors with `requires_grad` set receive gradients. Gradients are accumulated
         * rather than overwritten, so they should be reset to zero between training steps.
         *
         * @param root The tensor to differentiate, typically a scalar loss.
         */
        static void backward(std::shared_ptr<Tensor> root);
};

#endif