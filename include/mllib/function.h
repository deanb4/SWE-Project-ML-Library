#ifndef FUNCTION_
#define FUNCTION_

#include <string>
#include "tensor.h"

class Function {
	private:
		std::string op_type;
		std::shared_ptr<Tensor> inputs;
		std::shared_ptr<Tensor> saved_tensors;
	public:
		Tensor add(const Tensor& a, const Tensor& b); 
		Tensor sub(const Tensor& a, const Tensor& b);
		Tensor mul(const Tensor& a, const Tensor& b);
		Tensor div(const Tensor& a, const Tensor& b);
        Tensor matmul(const Tensor& a, const Tensor& b);
        Tensor sum(const Tensor& a);
        Tensor mean(const Tensor& a);
        Tensor exp(const Tensor& a);
        Tensor log(const Tensor& a);
        Tensor sqrt(const Tensor& a);
        Tensor tanh(const Tensor& a);
        Tensor relu(const Tensor& a);
        Tensor sigmoid(const Tensor& a);
        Tensor softmax(const Tensor& a, int dim);
        Tensor reshape(const Tensor& a, const std::vector<int>& shape);
        Tensor transpose(const Tensor& a, int dim0, int dim1);
        std::vector<Tensor> backward(const Tensor& grad_output);
};
#endif

