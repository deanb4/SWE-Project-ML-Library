#include "mllib/layernorm.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    //Check if input dimensions are valid
    void validate_dimension(size_t dim) {
        if (dim == 0) {
            throw std::invalid_argument("LayerNorm: dim must be greater than zero");
        }

        if (dim > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("LayerNorm: dim is too large to represent as a tensor dimension");
        }
    }

    //Create and initialize a parameter tensor
    Tensor create_parameter(size_t dim, float initial_value) {
        validate_dimension(dim);

        if (dim > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error("LayerNorm: parameter allocation size overflows size_t");
        }
        const size_t nbytes = dim * sizeof(float);

        //Currently only CPU supported MAKE NOTE OF THIS <------------------------------------------------------------------------------------
        auto storage = std::make_shared<Storage>(nbytes, Device::CPU);

        float* values = static_cast<float*>(storage->get_ptr());

        for (size_t i = 0; i < dim; ++i) {
            values[i] = initial_value;
        }

        // Parameters are learnable, so requires_grad is true.
        return Tensor(
            std::move(storage),
            {static_cast<int64_t>(dim)},
            Device::CPU,
            true,
            Dtype::FLOAT32
        );
    }

    /**
     * @brief Performs LayerNorm on a floating-point input tensor.
     *
     * Each consecutive group of `dim` elements is treated as one feature vector.
     * The function calculates that vector's mean and population variance,
     * normalizes it, and applies gamma and beta.
     *
     * @tparam T Floating-point data type (float or double).
     *
     * @param x Input tensor.
     * @param gamma Learned scale parameter (FLOAT32).
     * @param beta Learned shift parameter (FLOAT32).
     * @param eps Small constant added to the variance.
     * @param dim Number of features in each vector.
     *
     * @return A normalized tensor with the same shape and dtype as x.
     */
    template<typename T>
    Tensor layer_norm(
        const Tensor& x,
        const Tensor& gamma,
        const Tensor& beta,
        double eps,
        size_t dim
    ) {
        const size_t element_count = x.num_elements();

        if (element_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("LayerNorm: output allocation size overflows size_t");
        }

        //allocate memory for the output tensor.
        auto output_storage = std::make_shared<Storage>(element_count * sizeof(T), Device::CPU);

        //obtain pointers to the underlying tensor data.
        const T* input_values = static_cast<const T*>(x.get_data()->get_ptr());
        const float* gamma_values = static_cast<const float*>(gamma.get_data()->get_ptr());
        const float* beta_values = static_cast<const float*>(beta.get_data()->get_ptr());
        T* output_values = static_cast<T*>(output_storage->get_ptr());

        //each group of `dim` consecutive elements is one vector.
        const size_t vector_count = element_count / dim;

        for (size_t vector_index = 0; vector_index < vector_count; ++vector_index) {
            const size_t offset = vector_index * dim;

            //Calculate the mean
            //using double for accumulation to reduce rounding errors.
            double mean = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                mean += static_cast<double>(input_values[offset + j]);
            }
            mean /= static_cast<double>(dim);

            //Calculate population variance
            double variance = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                const double difference = static_cast<double>(input_values[offset + j]) - mean;
                variance += difference * difference;
            }
            variance /= static_cast<double>(dim);

            //Calculate standard deviation
            const double standard_deviation = std::sqrt(variance + eps);

            //Normalize, scale, and shift the vector
            for (size_t j = 0; j < dim; ++j) {
                const double value = static_cast<double>(input_values[offset + j]);
                const double normalized = (value - mean) / standard_deviation;
                const double scaled = static_cast<double>(gamma_values[j]) * normalized;
                const double shifted = scaled + static_cast<double>(beta_values[j]);
                output_values[offset + j] = static_cast<T>(shifted);
            }
        }

        return Tensor(
            std::move(output_storage),
            x.get_shape(),
            Device::CPU,
            x.get_requires_grad() ||
                gamma.get_requires_grad() ||
                beta.get_requires_grad(),
            x.get_dtype()
        );
    }

} //anonymous namespace with helper functions

LayerNorm::LayerNorm(size_t dim, double eps) 
    : gamma(create_parameter(dim, 1.0f)), beta(create_parameter(dim, 0.0f)), eps(eps) {

    //validate epsilon
    if (!std::isfinite(eps) || eps <= 0.0) {
        throw std::invalid_argument("LayerNorm: eps must be a finite value greater than zero");
    }
}

Tensor LayerNorm::forward(const Tensor& x) const {
    const std::vector<int64_t>& shape = x.get_shape();

    //a scalar has no last dimension to normalize
    if (shape.empty()) {
        throw std::invalid_argument("LayerNorm::forward: input must have at least one dimension");
    }

    //the last dimension must match the number of gamma/beta values.
    const int64_t last_dimension = shape.back();

    if (last_dimension <= 0) {
        throw std::invalid_argument("LayerNorm::forward: last input dimension must be greater than zero");
    }

    const size_t dim = static_cast<size_t>(last_dimension);

    const size_t parameter_dim = static_cast<size_t>(gamma.get_shape()[0]);

    if (dim != parameter_dim) {
        throw std::invalid_argument("LayerNorm::forward: input's last dimension does not match the LayerNorm dimension");
    }

    //currently support CPU only.
    if (x.get_device() != Device::CPU) {
        throw std::runtime_error("LayerNorm::forward: only CPU tensors are currently supported");
    }

    //layer normalization requires floating-point input.
    switch (x.get_dtype()) {
        case Dtype::FLOAT32:
            return layer_norm<float>(x, gamma, beta, eps, dim);

        case Dtype::FLOAT64:
            return layer_norm<double>(x, gamma, beta, eps, dim);

        default:
            throw std::invalid_argument("LayerNorm::forward: input must have FLOAT32 or FLOAT64 dtype");
    }
}

std::vector<Tensor> LayerNorm::parameters() const {
    return {gamma, beta};
}