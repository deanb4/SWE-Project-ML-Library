#include "mllib/tensor.h"
#include <stdexcept>
#include "mllib/autograd.h"

Tensor::Tensor(std::shared_ptr<Storage> data, std::vector<int64_t> shape,
                Device device, bool requires_grad = false, Dtype dtype) :
                data(std::move(data)), shape(std::move(shape)), device(device), requires_grad(requires_grad), dtype(dtype) {
    // Error checking

    // Check data 
    if (!this->data) {
        throw std::invalid_argument("Tensor: storage is null");
    }

    // Check for same device
    if (this->data->get_device() != this->device) {
        throw std::invalid_argument("Tensor: storage device does nto match tensor device");
    }

    // Check for negative dim
    for (int64_t dim : this->shape) {
        if (dim < 0) {
            throw std::invalid_argument("Tensor: negative dimension in shape");
        }
    }

}


size_t Tensor::num_elements() const{
    // product of the dim in shape. (empty shape equals 1)
    size_t n = 1;
    for (int64_t dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    return n;
}

void Tensor::backward() {
    if (!requires_grad) {
        throw std::runtime_error("backward() called on a tensor that does not require grad");
    }

    if (num_elements() != 1) {
        throw std::runtime_error("backward() only supported on scalar tensors");
    }

    // TODO: shallow copy: shares grad_fn, so autograd walks the same graph (Check if sufficient space wise)
    std::shared_ptr<Tensor> root = std::make_shared<Tensor>(*this);
    Autograd::backward(root);

    // autograd seeded the copy's grad, keep it on this tensor too
    grad = root->get_grad();
}

Tensor Tensor::to(Device target_device) const {
    if (target_device == device) {
        return *this; // // aliases the same storage, no copy TODO: validate 
    }

    // Copy to target device
    std::shared_ptr<Storage> new_storage = data->copy_to(target_device);

    return Tensor(std::move(new_storage), shape, target_device, requires_grad, dtype);

}

std::shared_ptr<Storage> Tensor::get_data() const {
    return data;
}

Device Tensor::get_device() const {
    return device;
}

Dtype Tensor::get_dtype() const {
    return dtype;
}

const std::vector<int64_t>& Tensor::get_shape() const{
    return shape;
}

bool Tensor::get_requires_grad() const {
    return requires_grad;
}
void Tensor::set_requires_grad(bool value) {
    requires_grad = value;
}

std::shared_ptr<Tensor> Tensor::get_grad() const{
    return grad;
}

void Tensor::set_grad(std::shared_ptr<Tensor>& value) {
    grad = std::move(value);
}

std::shared_ptr<Function> Tensor::get_grad_fn() const {
    return grad_fn;
}

 void Tensor::set_grad_fn(const std::shared_ptr<Function>& value) {
    grad_fn = std::move(value);
 }