// functions
#include "mllib/function.h"
#include <stdexcept>


// Anonymous namespace for Helper functions
namespace {

    template<typename T>
    void matmul_kernel(const void* a, const void* b, void* out, size_t M, size_t K, size_t N) {
        // multiply row by column (dot product) 
        const T* a_ptr = static_cast<const T*>(a);
        const T* b_ptr = static_cast<const T*>(b);
        T* out_ptr = static_cast<T*>(out);
        
        // matrix multiplication
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                }
                out_ptr[i * N + j] = sum;
            }
        }
 
    }

    template<typename T, typename Op>
    void binary_kernel(const void* a, const void* b, void* out, size_t n, Op op) {
        const T* a_ptr = static_cast<const T*>(a);
        const T* b_ptr = static_cast<const T*>(b);
        T* out_ptr = static_cast<T*>(out)

        for (size_t i = 0; i < n ; ++i) {
            out_ptr[i] = static_cast<T>(op(a_ptr[i], b_ptr[i]));
        }
    }

    template<typename Op>
    Tensor binary_op(const Tensor& a, const Tensor& b, const std::string& name, Op op) {
        // Add error checking
        if (a.get_shape() != b.get_shape()) {
            throw std::runtime_error("add: shape mismatch");
        }

        if (a.get_dtype() != b.get_dtype()) {
            throw std::runtime_error("add: dtype mismatch");
        }

        if (a.get_device() != b.get_device()) {
            throw std::runtime_error("add: device mismatch");
        }

        // allocate output storage
        Dtype dtype = a.get_dtype();
        size_t n = a.num_elements();
        auto out_storage = std::make_shared<Storage>(n * dtype_size(dtype), Device::CPU);

        const void* a_data = a.get_data()->get_ptr();
        const void* b_data = b.get_data()->get_ptr();
        void* out_data = out_storage->get_ptr();

        // add matrices according to dtype
        switch (dtype) {
            case Dtype::FLOAT64:
                binary_kernel<double>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::FLOAT32:
                binary_kernel<float>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::INT64:
                binary_kernel<int64_t>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::INT32:
                binary_kernel<int32_t>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::INT16:
                binary_kernel<int16_t>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::INT8:
                binary_kernel<int8_t>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::UINT8:
                binary_kernel<uint8_t>(a_data, b_data, out_data,n, op);
                break;
            case Dtype::BOOL:
                binary_kernel<bool>(a_data, b_data, out_data,n, op);
                break;
            
        }

        bool requires_grad = a.get_requires_grad() || b.get_requires_grad();

        return Tensor(out_storage, a.get_shape(), Device::CPU, requires_grad, dtype);
    }

}


Function::Function(std::string op_type, std::shared_ptr<Tensor> inputs, std::shared_ptr<Tensor> saved_tensors) :
                    op_type(op_type) , inputs(inputs), saved_tensors(saved_tensors) {}


Tensor Function::add(const Tensor& a, const Tensor& b) {
   return binary_op(a, b, "add", [](auto x, auto y) { return x + y; });
}

Tensor Function::sub(const Tensor& a, const Tensor& b) {
   return binary_op(a, b, "sub", [](auto x, auto y) { return x - y; });
}

Tensor Function::mul(const Tensor& a, const Tensor& b) {
   return binary_op(a, b, "mul", [](auto x, auto y) { return x * y; });
}


Tensor Function::div(const Tensor& a, const Tensor& b) {
   return binary_op(a, b, "div", [](auto x, auto y) { return x / y; });
}

Tensor Function::matmul(const Tensor& a, const Tensor& b) {
    if (a.get_dtype() != b.get_dtype()) {
        throw std::runtime_error("matmul: dtype mismatch");
    }

    if (a.get_device() != b.get_device()) {
        throw std::runtime_error("matmul: device mismatch");
    }

    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    if (a_shape.size() != 2 || b_shape.size() !=2) {
        throw std::runtime_error("matmul: both tensors must be 2D")
    }

    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("matmul: inner dimensions must match");
    }

    size_t M = static_cast<size_t>(a_shape[0]);
    size_t K = static_cast<size_t>(a_shape[1]);
    size_t N = static_cast<size_t>(b_shape[1]);

    Dtype dtype = a.get_dtype();
    auto out_storage = std::make_shared<Storage>(M * N * dtype_size(dtype), Device::CPU);

    const void* a_data = a.get_data()->get_ptr();
    const void* b_data = b.get_data()->get_ptr();
    void* out_data = out_storage->get_ptr();
    
    switch (dtype) {
            case Dtype::FLOAT64:
                matmul_kernel<double>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::FLOAT32:
                matmul_kernel<float>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::INT64:
                matmul_kernel<int64_t>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::INT32:
                matmul_kernel<int32_t>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::INT16:
                matmul_kernel<int16_t>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::INT8:
                matmul_kernel<int8_t>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::UINT8:
                matmul_kernel<uint8_t>(a_data, b_data, out_data, M, K, N);
                break;
            case Dtype::BOOL:
                matmul_kernel<bool>(a_data, b_data, out_data, M, K, N);
                break;
            
        }

}


