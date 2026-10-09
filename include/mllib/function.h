#ifndef FUNCTION_
#define FUNCTION_

#include <string>
#include <vector>

#ifndef TENSOR_H
    #include "tensor.h"
#endif


class Function {
	private:
		std::string op_type;
		std::shared_ptr<Tensor> inputs;
		std::shared_ptr<Tensor> saved_tensors;
	public:
        Function(std::string op_type, std::shared_ptr<Tensor> inputs, std::shared_ptr<Tensor> saved_tensors);
		Tensor add(const Tensor& a, const Tensor& b); 
		Tensor sub(const Tensor& a, const Tensor& b);
		Tensor mul(const Tensor& a, const Tensor& b);
		Tensor div(const Tensor& a, const Tensor& b);
        Tensor matmul(const Tensor& a, const Tensor& b);
        Tensor sum(const Tensor& a);
        Tensor mean(const Tensor& a);

        // Josh ******************************** (Note* any issues just message me)
        Tensor exp(const Tensor& a);
        Tensor log(const Tensor& a);
        Tensor sqrt(const Tensor& a);
        Tensor tanh(const Tensor& a);
        Tensor relu(const Tensor& a);
        Tensor sigmoid(const Tensor& a);
        Tensor softmax(const Tensor& a, int dim);
        // Josh ********************************

        Tensor reshape(const Tensor& a, const std::vector<int>& shape);
        Tensor transpose(const Tensor& a, int dim0, int dim1);
        std::vector<Tensor> backward(const Tensor& grad_output);
};
#endif

