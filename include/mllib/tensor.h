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
#include "types.h"


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

        Tensor to(Device device) const; // move to device

        // getters / setters
        std::shared_ptr<Storage> get_data() const;
        Device get_device() const;
        Dtype get_dtype() const;
        const std::vector<int64_t>& get_shape() const;
    
        bool get_requires_grad() const;
        void set_requires_grad(bool value);

        std::shared_ptr<Tensor> get_grad() const;
        void set_grad(std::shared_ptr<Tensor> value);

        std::shared_ptr<Function> get_grad_fn() const;
        void set_grad_fn(std::shared_ptr<Function> value);
        

};


#endif