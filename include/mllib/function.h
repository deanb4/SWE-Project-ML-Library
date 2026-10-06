#ifndef FUNCTION_
#define FUNCTION_

#include "tensor.h"

class Function {
	private:
		string op_type;
		list inputs;
		list saved_tensors;
	public:
		Tensor add(float a, float b);
		Tensor sub(float a, float b);
		Tensor mul(float a, float b);
		Tensor div(float a, float b);
		// ...
};
#endif
