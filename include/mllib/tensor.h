/* 
Header for Tensor class
*/
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>
#include <memory>
#include "storage.h"
#include "types.h"

// #ifndef FUNCTION_
//     #include "function.h"
// #endif
class Function; // forward declaration 


class Tensor {
    private:
        std::shared_ptr<Storage> data; // Need to create class
        Device device;
        Dtype dtype;
        std::vector<int64_t> shape;
        bool requires_grad;
        std::shared_ptr<Function> grad_fn; // need to create class

        // every copy of a tensor shares one GradSlot, so a grad written through 
        // any copy (like the ones the graph keeps) show up on all of them
        struct GradSlot { 
            std::shared_ptr<Tensor> grad;
        };
        std::shared_ptr<GradSlot> grad_slot;

    public:
        // constructor
        Tensor(std::shared_ptr<Storage> data, std::vector<int64_t> shape,
                Device device = Device::CPU, bool requires_grad = false, Dtype dtype = Dtype::FLOAT32);

        // new tensor with every element set to value (FLOAT32 / FLAOT64 only)
        static Tensor full(const std::vector<int64_t>& shape, double value, Dtype dtype);
        
        // backward
        void backward();

        Tensor to(Device target_device) const; // move to device

        // getters / setters
        std::shared_ptr<Storage> get_data() const;
        Device get_device() const;
        Dtype get_dtype() const;
        const std::vector<int64_t>& get_shape() const;
    
        bool get_requires_grad() const;
        void set_requires_grad(bool value);

        std::shared_ptr<Tensor> get_grad() const;
        void set_grad(std::shared_ptr<Tensor>& value);

        std::shared_ptr<Function> get_grad_fn() const;
        void set_grad_fn(const std::shared_ptr<Function>& value);

        size_t num_elements() const;
        

};


#endif