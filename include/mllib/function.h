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
};
#endif
