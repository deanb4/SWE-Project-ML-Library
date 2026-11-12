// functions
#include "mllib/function.h"
#include "mllib/autograd.h"
#include <cstring>
#include <stdexcept>


// Anonymous namespace for Helper functions
namespace {

    template<typename T>
    void mean_kernel(const void* a, void* out, size_t N) {
        const T* a_ptr = static_cast<const T*>(a);
        T* out_ptr = static_cast<T*>(out);
        double sum = 0;
        for (size_t i = 0; i < N; ++i) {
            sum += a_ptr[i];
        }
        double mean = sum / static_cast<double>(N);
        out_ptr[0] = static_cast<T>(mean);
    }

    template<typename T>
    void sum_kernel(const void* a, void* out, size_t N) {
        const T* a_ptr = static_cast<const T*>(a);
        T* out_ptr = static_cast<T*>(out);
        double sum = 0;
        for (size_t i = 0; i < N; ++i) {
            sum += a_ptr[i];
        }
        out_ptr[0] = static_cast<T>(sum);
    }

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
        T* out_ptr = static_cast<T*>(out);

        for (size_t i = 0; i < n ; ++i) {
            out_ptr[i] = static_cast<T>(op(a_ptr[i], b_ptr[i]));
        }
    }

    // Builds the graph: remembers which op made "out" and from which inputs, so
    // backward can walk from out back to them. Skipped when no grad is needed.
    void record(Tensor& out, const std::string& op_type, const std::vector<Tensor>& inputs,
                std::vector<int> saved_dims = {}) {
                    
        if (!out.get_requires_grad() || !Autograd::is_grad_enabled()) {
            return;
        }

        std::vector<std::shared_ptr<Tensor>> input_ptrs;
        for (const Tensor& input : inputs) {
            input_ptrs.push_back(std::make_shared<Tensor>(input));
        }
        out.set_grad_fn(std::make_shared<Function>(op_type, input_ptrs, nullptr, saved_dims));
    }

    // The value of a one-element float tensor (the grad of sum and mean's output)
    double scalar_value(const Tensor& t) {
        if (t.get_dtype() == Dtype::FLOAT32) {
            return *static_cast<const float*>(t.get_data()->get_ptr());
        }
        if (t.get_dtype() == Dtype::FLOAT64) {
            return *static_cast<const double*>(t.get_data()->get_ptr());
        }
        throw std::runtime_error("backward: only float tensors have gradients");
    }
    

    template<typename Op>
    Tensor binary_op(const Tensor& a, const Tensor& b, const std::string& name, Op op) {
        // Add error checking
        if (a.get_shape() != b.get_shape()) {
            throw std::runtime_error(name + ": shape mismatch");
        }

        if (a.get_dtype() != b.get_dtype()) {
            throw std::runtime_error(name + ": dtype mismatch");
        }

        if (a.get_device() != b.get_device()) {
            throw std::runtime_error(name + ": device mismatch");
        }

        // allocate output storage
        Dtype dtype = a.get_dtype();
        size_t n = a.num_elements();
        auto out_storage = std::make_shared<Storage>(n * dtype_size(dtype), Device::CPU);

        const void* a_data = a.get_data()->get_ptr(); // data tensor 1
        const void* b_data = b.get_data()->get_ptr(); // data tensor 2
        void* out_data = out_storage->get_ptr(); // where data goes for answer

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
                throw std::runtime_error("bool tensors not supported");
            
        }

        bool requires_grad = a.get_requires_grad() || b.get_requires_grad();

        Tensor out(out_storage, a.get_shape(), Device::CPU, requires_grad, dtype);
        record(out, name, {a,b});
        return out;
    }

}


Function::Function(std::string op_type, std::vector<std::shared_ptr<Tensor>> inputs, 
                    std::shared_ptr<Tensor> saved_tensors, std::vector<int> saved_dims) :
                    op_type(op_type), inputs(inputs), 
                    saved_tensors(saved_tensors), saved_dims(saved_dims) {}

const std::vector<std::shared_ptr<Tensor>>& Function::get_inputs() const {
    return inputs;
}

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
        throw std::runtime_error("matmul: both tensors must be 2D");
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
                throw std::runtime_error("matmul: bool tensors not supported");
        }

        bool requires_grad = a.get_requires_grad() || b.get_requires_grad();

        Tensor out(out_storage, {a_shape[0],b_shape[1]}, Device::CPU, requires_grad, dtype);
        record(out, "matmul", {a, b});
        return out;
}



Tensor Function::sum(const Tensor& a) {
    Dtype dtype = a.get_dtype();
    size_t N = a.num_elements();
    std::shared_ptr<Storage> out_storage = std::make_shared<Storage>(dtype_size(dtype), Device::CPU);
    const void* a_data = a.get_data()->get_ptr();
    void* out_data = out_storage->get_ptr();

    // Check for empty Tensor (data)
    if (N == 0) {
        throw std::runtime_error("Empty Tensor");
    }

    switch (dtype) {
        case Dtype::FLOAT64:
            sum_kernel<double>(a_data, out_data, N);
            break;
        case Dtype::FLOAT32:
            sum_kernel<float>(a_data, out_data, N);
            break;
        case Dtype::INT64:
            sum_kernel<int64_t>(a_data, out_data,N);
            break;
        case Dtype::INT32:
            sum_kernel<int32_t>(a_data, out_data, N);
            break;
        case Dtype::INT16:
            sum_kernel<int16_t>(a_data, out_data,N);
            break;
        case Dtype::INT8:
            sum_kernel<int8_t>(a_data, out_data, N);
            break;
        case Dtype::UINT8:
            sum_kernel<uint8_t>(a_data, out_data, N);
            break;
        case Dtype::BOOL:
            throw std::runtime_error("sum: bool tensors not supported");
    }

    bool requires_grad = a.get_requires_grad();

    Tensor out(out_storage, {1}, Device::CPU, requires_grad, dtype);
    record(out, "sum", {a});
    return out;

}

// For now keeping float only due to division issue/error
Tensor Function::mean(const Tensor& a) {
    Dtype dtype = a.get_dtype();
    size_t N = a.num_elements();
    std::shared_ptr<Storage> out_storage = std::make_shared<Storage>(dtype_size(dtype), Device::CPU);
    const void* data = a.get_data()->get_ptr();
    void* out_data = out_storage->get_ptr();

    // Check for division by zero
    if (N == 0) {
        throw std::runtime_error("division by zero");
    }

    switch (dtype) {
        case Dtype::FLOAT64:
            mean_kernel<double>(data, out_data, N);
            break;
        case Dtype::FLOAT32:
            mean_kernel<float>(data, out_data, N);
            break;
        case Dtype::INT64:
            // mean_kernel<int64_t>(data, out_data, N);
            throw std::runtime_error("mean: only supports floats");
            break;
        case Dtype::INT32:
            // mean_kernel<int32_t>(data, out_data, N);
            throw std::runtime_error("mean: only supports floats");
            break;
        case Dtype::INT16:
            // mean_kernel<int16_t>(data, out_data, N);
            throw std::runtime_error("mean: only supports floats");
            break;
        case Dtype::INT8:
            // mean_kernel<int8_t>(data, out_data, N);
            throw std::runtime_error("mean: only supports floats");
            break;
        case Dtype::UINT8:
            // mean_kernel<uint8_t>(data, out_data, N);
            throw std::runtime_error("mean: only supports floats");
            break;
        case Dtype::BOOL:
            throw std::runtime_error("mean: bool not supported");
            break;
    }

    bool requires_grad = a.get_requires_grad();
    Tensor out(out_storage, {1}, Device::CPU, requires_grad, dtype);
    record(out, "mean", {a});
    return out;

}


Tensor Function::reshape(const Tensor& a, const std::vector<int64_t>& shape) {
    int64_t num_elements = static_cast<int64_t>(a.num_elements());

    // new shape array
    std::vector<int64_t> new_shape = shape;
    // error checking
    int64_t known = 1;
    bool has_infer = false;
    size_t infer_idx = 0;

    // single pass over dim
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (has_infer) {
                throw std::runtime_error("reshape: only one dimnesion can be -1");
            }

            has_infer = true;
            infer_idx = i;
        } else if (new_shape[i] < 0) {
            throw std::runtime_error("reshape: invalid dimension " + std::to_string(new_shape[i]));
        } else {
            known *= new_shape[i];
        }
    }

    if (has_infer) {
        if (known == 0) {
            throw std::runtime_error("reshape: cannot infer -1 when antoher dimension is 0");
        }

        new_shape[infer_idx] = num_elements / known;

        if (num_elements % known != 0) {
            throw std::runtime_error("reshape: shape is invalid for input of size" + 
                                    std::to_string(num_elements));
        }


    } else if (known != num_elements) {
        throw std::runtime_error("reshape: shape is invalid for input of size " + std::to_string(num_elements));
    }

    // return tensor
    Tensor out(a.get_data(), new_shape, a.get_device(), 
                  a.get_requires_grad(), a.get_dtype());

    // to build graph to be used for back propagation
    record(out, "reshape", {a});

    return out;
}

Tensor Function::transpose(const Tensor& a, int dim0, int dim1) {
    const std::vector<int64_t>& shape = a.get_shape();
    int ndim = static_cast<int>(shape.size());

    // negative dims count from the end 
    if (dim0 < 0) {
        dim0 += ndim;
    }
    if (dim1 < 0) {
        dim1 += ndim;
    }
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::runtime_error("transpose: dim out of range");
    }
    
    std::vector<int64_t> out_shape = shape;
    std::swap(out_shape[dim0], out_shape[dim1]); 

    // strides[d] = how far apart (in elements) two neighbors along dim d are in the tensor
    std::vector<size_t> strides(ndim);
    size_t stride = 1;
    for (int d = ndim -1; d >= 0; d--) {
        strides[d] = stride;
        stride *= static_cast<size_t>(shape[d]);
    }

    std::swap(strides[dim0], strides[dim1]);

    size_t n = a.num_elements();
    size_t elem_size = dtype_size(a.get_dtype());
    auto out_storage = std::make_shared<Storage>(n * elem_size, Device::CPU);
    const char* src = static_cast<const char*>(a.get_data()->get_ptr());
    char* dst = static_cast<char*>(out_storage->get_ptr());

    for (size_t i = 0; i < n; i++) {
        // split output index i into position along each dim
        size_t rest = i;
        size_t src_index = 0;
        for (int d = ndim -1; d >=0; d--) {
            size_t dim_size = static_cast<size_t>(out_shape[d]);
            src_index += (rest % dim_size) * strides[d];
            rest /= dim_size;
        }

        // copy the elemtne's bytes, so every dtype works without a switch
        std::memcpy(dst + i * elem_size, src + src_index * elem_size, elem_size);
    }

    Tensor out(out_storage, out_shape, Device::CPU, a.get_requires_grad(), a.get_dtype());
    record(out, "transpose", {a}, {dim0, dim1});
    return out;
}

// Given the gradient of the loss w.r.t. this op's output, returns the gradient
// w.r.t each input (same order as inputs)
std::vector<Tensor> Function::backward(const Tensor& grad_output) {
    const Tensor& g = grad_output;
    const Tensor& a = *inputs[0];

    if (op_type == "add") {
        return {g, g};
    }
    if (op_type == "sub") {
        Tensor zeros = Tensor::full(g.get_shape(), 0.0, g.get_dtype());
        return {g, sub(zeros,g)};
    }
    if (op_type == "mul") {
        const Tensor& b = *inputs[1];
        return {mul(g,b), mul(g,a)};
    }
    if (op_type == "div") {
        // d(a/b)/da = 1/b, d(ab)/db = -a/b^2
        const Tensor&b = *inputs[1];
        Tensor zeros = Tensor::full(b.get_shape(), 0.0, b.get_dtype());
        return {div(g,b), sub(zeros, div(mul(g,a), mul(b,b)))};
    }
    if (op_type == "matmul") {
        // out = a @ b | d/da = g@ b^T, d/db == a^T @ g
        const Tensor& b = *inputs[1];
        return {matmul(g, transpose(b,0,1)), matmul(transpose(a,0,1),g)};
    }
    if (op_type == "sum") {
        // every element adds to get the sum once, so each one gets g
        return {Tensor::full(a.get_shape(), scalar_value(g), a.get_dtype())};
    }
    if (op_type == "mean") {
        // every element adds 1/N of itself to the mean
        double value = scalar_value(g) / static_cast<double>(a.num_elements());
        return {Tensor::full(a.get_shape(), value, a.get_dtype())};
    }
    if (op_type == "reshape") {
        return {reshape(g, a.get_shape())};
    }
    if (op_type == "transpose") {
        // swapping the same two dims again undoes it
        return {transpose(g, saved_dims[0], saved_dims[1])};
    }

    throw std::runtime_error("backward not implemented for " + op_type);
}

//   Tensor reshape(const Tensor& a, const std::vector<int>& shape);
//         Tensor transpose(const Tensor& a, int dim0, int dim1);
//         std::vector<Tensor> backward(const Tensor& grad_output);