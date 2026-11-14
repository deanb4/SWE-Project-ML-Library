// Trains a 2 -> 32 -> 1 netowrk to learn XOR, the classic problme a model without
// a hidden layer and nonlinearity cannon solve

#include <cstdio>
#include <cstring>
#include <vector>
#include "mllib/function.h"
#include "mllib/linear.h"
#include "mllib/optimizer.h"
#include "mllib/tensor.h"


// FLOAT32 tensor holding 'values'
Tensor make_tensor(const std::vector<float>& values, std::vector<int64_t> shape) {
    auto storage = std::make_shared<Storage>(values.size() * sizeof(float), Device::CPU);
    std::memcpy(storage->get_ptr(), values.data(), values.size() * sizeof(float));
    return Tensor(storage, shape, Device::CPU, false, Dtype::FLOAT32);
}

int main() {
    Function F("trainh", {}, nullptr);

    // every XOR input, one per row, and the answer for each
    Tensor x = make_tensor({0,0,
                            0,1,
                            1,0,
                            1,1}, {4, 2});
    Tensor y = make_tensor({0,1,1,0}, {4,1});

    Linear layer1(2,32);
    Linear layer2(32,1);

    std::vector<Tensor> params = layer1.parameters();
    for (const Tensor& p : layer2.parameters()) {
        params.push_back(p);
    }
    SGD optimizer(params, 0.05f);

    for (int step = 0; step<= 1000; step++) {
        Tensor hidden = F.relu(layer1.forward(x));
        Tensor pred = layer2.forward(hidden);

        // mean squared error
        Tensor diff = F.sub(pred,y);
        Tensor loss = F.mean(F.mul(diff,diff));

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (step % 100 == 0) {
            std::printf("step %4d loss %.6f\n", step, *static_cast<float*>(loss.get_data()->get_ptr()));
        }
    }
}