#include "mllib/conv2d.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    //Multiplies two sizes after checking for overflow
    size_t checked_multiply(size_t a, size_t b, const char* message) {
        if((b != 0) && (a > std::numeric_limits<size_t>::max() / b)) {
            throw std::overflow_error(message);
        }
        return a * b;
    }

    //Tensor dimensions are stored as int64_t, so each dimension must fit
    void check_tensor_dimension(size_t value, const char* name) {
        if(static_cast<uintmax_t>(value) > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument(std::string("Conv2d: ") + name +
                " is too large to represent as a tensor dimension");
        }
    }

    //Check if inputs for constructor are valid
    void validate_configuration(
        size_t in_channels, size_t out_channels, size_t kernel_size, int stride, int padding) {

        if(in_channels == 0) {
            throw std::invalid_argument("Conv2d: in_channels must be greater than zero");
        }
        if(out_channels == 0) {
            throw std::invalid_argument("Conv2d: out_channels must be greater than zero");
        }
        if(kernel_size == 0) {
            throw std::invalid_argument("Conv2d: kernel_size must be greater than zero");
        }
        if(stride <= 0) {
            throw std::invalid_argument("Conv2d: stride must be greater than zero");
        }
        if(padding < 0) {
            throw std::invalid_argument("Conv2d: padding must be non-negative");
        }

        check_tensor_dimension(in_channels, "in_channels");
        check_tensor_dimension(out_channels, "out_channels");
        check_tensor_dimension(kernel_size, "kernel_size");

        //Calculate the required areas safely
        const size_t kernel_area = checked_multiply(
            kernel_size, kernel_size, "Conv2d: kernel area overflows size_t");
        const size_t fan_in = checked_multiply(
            in_channels, kernel_area, "Conv2d: fan_in overflows size_t");
        const size_t weight_count = checked_multiply(
            out_channels, fan_in, "Conv2d: weight element count overflows size_t");
        (void)checked_multiply(
            weight_count, sizeof(float), "Conv2d: weight allocation size overflows size_t");
        (void)checked_multiply(
            out_channels, sizeof(float), "Conv2d: bias allocation size overflows size_t");
    }


    //Creates and initializes the learnable convolution kernels
    Tensor create_weight(
        size_t in_channels, size_t out_channels, size_t kernel_size, int stride, int padding) {
        
        //Validate configuration before allocating memory
        validate_configuration(in_channels, out_channels, kernel_size, stride, padding);

        const size_t kernel_area = kernel_size * kernel_size;
        const size_t fan_in = in_channels * kernel_area;
        const size_t weight_count = out_channels * fan_in;
        const size_t nbytes = weight_count * sizeof(float);

        //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
        auto storage = std::make_shared<Storage>(nbytes, Device::CPU);

        float* values = static_cast<float*>(storage->get_ptr());

        //Kaiming (He) normal initialization
        //fan_in = in_channels * kernel_size * kernel_size
        //standard deviation = sqrt(2 / fan_in)
        const float stddev = static_cast<float>(std::sqrt(2.0 / static_cast<double>(fan_in)));

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<float> distribution(0.0f, stddev);

        for(size_t i = 0; i < weight_count; ++i) {
            values[i] = distribution(generator);
        }

        //Create the weight tensor with shape (out_channels, in_channels, kernel_size, kernel_size)
        //requires_grad = true because weight is learnable
        return Tensor(
            std::move(storage),
            {
                static_cast<int64_t>(out_channels),
                static_cast<int64_t>(in_channels),
                static_cast<int64_t>(kernel_size),
                static_cast<int64_t>(kernel_size)
            },
            Device::CPU,
            true,
            Dtype::FLOAT32
        );
    }

    //Creates and initializes (to zero) the learnable bias tensor
    Tensor create_bias(size_t out_channels) {
        if(out_channels == 0) {
            throw std::invalid_argument("Conv2d: out_channels must be greater than zero");
        }

        check_tensor_dimension(out_channels, "out_channels");

        const size_t nbytes = checked_multiply(
            out_channels, sizeof(float), "Conv2d: bias allocation size overflows size_t");
        
        auto storage = std::make_shared<Storage>(nbytes, Device::CPU);

        float* values = static_cast<float*>(storage->get_ptr());

        //Initialize every bias value to zero
        std::fill_n(values, out_channels, 0.0f);

        //Create the bias tensor with shape (out_channels,)
        //requires_grad is true because bias is learnable
        return Tensor(
            std::move(storage),
            {static_cast<int64_t>(out_channels)},
            Device::CPU,
            true,
            Dtype::FLOAT32
        );
    }

}//Anonymous namespace with helper functions

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, int stride, int padding)
    : weight(create_weight(in_channels, out_channels, kernel_size, stride, padding)), 
    bias(create_bias(out_channels)),
    stride(stride),
    padding(padding) { }

Tensor Conv2d::forward(const Tensor& x) const {
    const std::vector<int64_t>& w_shape = weight.get_shape();
    const std::vector<int64_t>& x_shape = x.get_shape();
    if (x_shape.size() != 4 || x_shape[1] != w_shape[1]) {
        throw std::invalid_argument("Conv2d: input must have shape (N, in_channels, H, W)");
    }

    Function F("conv2d", {}, nullptr);
    const int64_t n = x_shape[0];
    const int64_t out_channels = w_shape[0];
    const int kernel = static_cast<int>(w_shape[2]);

    // Every window becomes one row: (N * H_out * W_out, in_channels * k * k) 
    Tensor cols = F.im2col(x, kernel, stride, padding);

    // each kernel flattened into one column: (in_channels * k * k, out_channels)
    Tensor w = F.transpose(F.reshape(weight, {out_channels, -1}), 0, 1);

    // one matmul applies every kernel to every window: (N * H_out * W_out, out_channels)
    Tensor out = F.matmul(cols, w);

    // add each channel's bias to every row: (rows, 1) ones @ (1, out_channels) bias, as in linear
    Tensor ones = Tensor::full({out.get_shape()[0], 1}, 1.0, Dtype::FLOAT32);
    out = F.add(out, F.matmul(ones, F.reshape(bias, {1, out_channels})));

    // rows are ordered (n, oh, ow), so out is really (N, H_out * W_out, out_channels);
    // move the channels in front of the positions to get (N, out_channels, H_out, W_out)
    const int64_t h_out =  (x_shape[2] + 2 * padding - kernel) / stride + 1;
    const int64_t w_out = (x_shape[3] + 2 * padding - kernel) / stride + 1;
    out = F.transpose(F.reshape(out, {n, h_out * w_out, out_channels}), 1,2);
    return F.reshape(out, {n, out_channels, h_out, w_out});
    
}

std::vector<Tensor> Conv2d::parameters() const {
    // Tensor copies share underlying Storage through shared_ptr
    return {weight, bias};
}