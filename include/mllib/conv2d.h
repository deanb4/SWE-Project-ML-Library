#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <cstddef>
#include "tensor.h"

/**
 * @class Conv2d
 * @brief A two-dimensional convolution layer with learnable weights and bias.
 *
 * The Conv2d class applies a set of learnable two-dimensional convolution kernels to an 
 * input tensor.
 *
 * For an input with `in_channels` channels and a convolution with `out_channels` kernels,
 * each kernel has the shape:
 * (in_channels, kernel_size, kernel_size)
 *
 * The complete weight tensor therefore has the shape:
 * (out_channels, in_channels, kernel_size, kernel_size)
 *
 * Each output channel has its own kernel and a corresponding scalar bias.
 *
 * For each output position `(i, j)`, the convolution computes a weighted sum over every input
 * channel and every position in the kernel window, then adds the bias associated with that 
 * output channel.
 *
 * Positions that fall outside the input due to padding are treated as zero.
 *
 * The output spatial dimensions are determined by:
 *
 *     H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
 *
 *     W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
 *
 * where `H` and `W` are the input height and width.
 *
 * The number of learnable parameters is:
 *
 *     out_channels * in_channels * kernel_size^2 + out_channels
 *
 * The weights should be initialized using Kaiming (He) initialization:
 *
 *     std(W) = sqrt(2 / fan_in)
 *
 * where:
 *
 *     fan_in = in_channels * kernel_size^2
 *
 * The bias is initialized to zero.
 *
 * The actual convolution operation in `forward()` is intended to be implemented using the
 * project's im2col-based convolution operation.
 */
class Conv2d {
    private:

        /**
         * @brief Learnable convolution kernels.
         *
         * Each output channel has one kernel. The tensor has shape:
         *
         *     (out_channels, in_channels, kernel_size, kernel_size)
         *
         * The first dimension selects which output channel the kernel produces, while the 
         * second dimension selects the input channel that the kernel operates on.
         */
        Tensor weight;

        /**
         * @brief Learnable bias for each output channel.
         *
         * There is one scalar bias value for each output channel.
         * The tensor has shape:
         *
         *     (out_channels)
         */
        Tensor bias;

        /**
         * @brief Number of elements by which the kernel moves for each step in the input.
         *
         * A stride of 1 moves the kernel one position at a time, a stride of 2 moves the kernel 
         * two positions at a time, etc.
         */
        int stride;

        /**
         * @brief Amount of zero-padding added around the spatial dimensions of the input.
         *
         * Padding of 0 means no padding is added. A padding of 1 adds one row of zeros above and
         * below the input and one column of zeros to the left and right.
         */
        int padding;

    public:

        /**
         * @brief Constructs a two-dimensional convolution layer.
         *
         * Creates the learnable weight and bias tensors and initializes them appropriately.
         *
         * The weight tensor has shape:
         *
         *     (out_channels, in_channels, kernel_size, kernel_size)
         *
         * and the bias tensor has shape:
         *
         *     (out_channels)
         *
         * The weights should be initialized using Kaiming (He) initialization with:
         *
         *     fan_in = in_channels * kernel_size^2
         *
         * and:
         *
         *     std(W) = sqrt(2 / fan_in)
         *
         * The bias should be initialized to zero.
         *
         * The constructor must reject invalid configurations. In particular, `in_channels`,
         * `out_channels`, `kernel_size`, and `stride` must be greater than zero, while `padding`
         * must be greater than or equal to zero.
         *
         * @param in_channels Number of channels in the input tensor.
         * @param out_channels Number of channels produced by the convolution.
         * @param kernel_size Height and width of each square convolution kernel.
         * @param stride Number of spatial positions the kernel moves per step.
         * @param padding Number of zero-padding elements added around the input.
         *
         * @throws std::invalid_argument If any channel count, kernel size, or stride is zero
         *         or negative, or if padding is negative.
         */
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, int stride, int padding);

        /**
         * @brief Applies the convolution to an input tensor.
         *
         * The forward pass computes a two-dimensional convolution using the
         * layer's weight and bias tensors.
         *
         * The spatial output dimensions are determined by:
         *
         *     H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
         *
         *     W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
         *
         * @param x Input tensor containing the data to convolve.
         * @return A tensor containing the convolution output.
         */
        Tensor forward(const Tensor& x) const;

        /**
         * @brief Returns the learnable parameters of this convolution layer.
         *
         * Conv2d has two learnable parameters: weight and bias.
         *
         * The weight tensor has shape:
         *
         *     (out_channels, in_channels, kernel_size, kernel_size)
         *
         * and the bias tensor has shape:
         *
         *     (out_channels)
         *
         * @return A vector containing the weight and bias tensors.
         */
        std::vector<Tensor> parameters() const;
};

#endif
