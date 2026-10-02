/* 
Header for Tensor class
*/
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>
#include <memory>
#include "storage.h"
#include "function.h"


enum class Device {
    CPU,
    GPU,
};

enum class Dtype {
    FLOAT64,
    FLOAT32,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL,
};

class Tensor {
    private:
        std::shared_ptr<Storage> data; // Need to create class
        Device device;
        Dtype dtype;
        std::vector<int64_t> shape;
        bool requires_grad;
        std::shared_ptr<Function> grad_fn; // need to create class
        std::shared_ptr<Tensor> grad;
        

    public:
        // constructor
        Tensor(std::shared_ptr<Storage> data, std::vector<int64_t> shape,
                Device device = Device::CPU, bool requires_grad = false, Dtype dtype = Dtype::FLOAT32);
        
        // backward
        void backward();

        Tensor to(Device device) const;

        // getters / setters
        std::shared_ptr<Storage> get_data() const { return data; }
        Device get_device() const { return device; }
        Dtype get_dtype() const { return dtype; }
        const std::vector<int64_t>& get_shape() const { return shape; }
    
        bool get_requires_grad() const { return requires_grad; }
        void set_requires_grad(bool value);

        std::shared_ptr<Tensor> get_grad() const { return grad; }
        void set_grad(std::shared_ptr<Tensor> value) { grad = value; }

        std::shared_ptr<Function> get_grad_fn() const { return grad_fn; }
        void set_grad_fn(std::shared_ptr<Function> value) { grad_fn = value; }
        

};


#endif