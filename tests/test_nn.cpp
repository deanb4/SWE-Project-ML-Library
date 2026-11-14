// Unit tests for the pieces of a neural network: relu, the Linear layer and the SGD optimizer.

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "mllib/function.h"
#include "mllib/linear.h"
#include "mllib/optimizer.h"
#include "mllib/storage.h"
#include "mllib/tensor.h"

namespace {

template<typename T>
Tensor make_tensor(const std::vector<T>& values, std::vector<int64_t> shape,
                   Dtype dtype, bool requires_grad = false) {
    auto storage = std::make_shared<Storage>(values.size() * sizeof(T), Device::CPU);
    if (!values.empty()) {
        std::memcpy(storage->get_ptr(), values.data(), values.size() * sizeof(T));
    }
    return Tensor(storage, std::move(shape), Device::CPU, requires_grad, dtype);
}

template<typename T>
std::vector<T> to_vector(const Tensor& t) {
    const T* ptr = static_cast<const T*>(t.get_data()->get_ptr());
    return std::vector<T>(ptr, ptr + t.num_elements());
}

std::vector<float> grad_of(const Tensor& t) {
    if (!t.get_grad()) {
        throw std::runtime_error("grad_of: tensor has no grad");
    }
    return to_vector<float>(*t.get_grad());
}

// Overwrite a tensor's values in place (to give a Linear layer known weights)
void set_values(const Tensor& t, const std::vector<float>& values) {
    std::memcpy(t.get_data()->get_ptr(), values.data(), values.size() * sizeof(float));
}

Function F{"test", {}, nullptr};

}  // namespace

// ---------------------------------------------------------------------------
// relu
// ---------------------------------------------------------------------------

TEST(Relu, ZeroesNegativesAndKeepsPositives) {
    Tensor a = make_tensor<float>({-2, -0.5f, 0, 0.5f, 3}, {5}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.relu(a)), (std::vector<float>{0, 0, 0, 0.5f, 3}));
}

TEST(Relu, KeepsShapeAndFloat64) {
    Tensor a = make_tensor<double>({-1, 2, -3, 4}, {2, 2}, Dtype::FLOAT64);
    Tensor out = F.relu(a);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(to_vector<double>(out), (std::vector<double>{0, 2, 0, 4}));
}

TEST(Relu, IntegerTensorThrows) {
    Tensor a = make_tensor<int32_t>({1, -1}, {2}, Dtype::INT32);
    EXPECT_THROW(F.relu(a), std::runtime_error);
}

TEST(Relu, GradientPassesOnlyWhereInputWasPositive) {
    // at exactly 0 the gradient is 0
    Tensor a = make_tensor<float>({-2, 0, 1, 3}, {4}, Dtype::FLOAT32, true);
    Tensor w = make_tensor<float>({10, 20, 30, 40}, {4}, Dtype::FLOAT32);
    F.sum(F.mul(F.relu(a), w)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{0, 0, 30, 40}));
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

TEST(Linear, ParametersHaveExpectedShapes) {
    Linear layer(2, 3);
    std::vector<Tensor> params = layer.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].get_shape(), (std::vector<int64_t>{2, 3})); // weight
    EXPECT_EQ(params[1].get_shape(), (std::vector<int64_t>{1, 3})); // bias
    EXPECT_TRUE(params[0].get_requires_grad());
    EXPECT_TRUE(params[1].get_requires_grad());
}

TEST(Linear, BiasStartsAtZero) {
    Linear layer(4, 3);
    EXPECT_EQ(to_vector<float>(layer.parameters()[1]), (std::vector<float>{0, 0, 0}));
}

TEST(Linear, ForwardComputesInputTimesWeightPlusBias) {
    Linear layer(2, 3);
    set_values(layer.parameters()[0], {1, 2, 3,
                                       4, 5, 6});
    set_values(layer.parameters()[1], {0.5f, -1, 2});

    // the identity as input picks out the rows of the weight
    Tensor x = make_tensor<float>({1, 0,
                                   0, 1}, {2, 2}, Dtype::FLOAT32);
    Tensor out = layer.forward(x);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{1.5f, 1, 5,
                                                         4.5f, 4, 8}));
}

TEST(Linear, BackwardGivesWeightAndBiasGradients) {
    // loss = sum(x @ W + b): dW[i][j] = sum of column i of x, db[j] = number of rows
    Linear layer(2, 3);
    Tensor x = make_tensor<float>({1, 2,
                                   3, 4}, {2, 2}, Dtype::FLOAT32);
    F.sum(layer.forward(x)).backward();

    EXPECT_EQ(grad_of(layer.parameters()[0]), (std::vector<float>{4, 4, 4,
                                                                  6, 6, 6}));
    EXPECT_EQ(grad_of(layer.parameters()[1]), (std::vector<float>{2, 2, 2}));
}

TEST(Linear, NonMatrixInputThrows) {
    Linear layer(2, 3);
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32);
    EXPECT_THROW(layer.forward(x), std::invalid_argument);
}

TEST(Linear, ZeroSizeThrows) {
    EXPECT_THROW(Linear(0, 3), std::invalid_argument);
    EXPECT_THROW(Linear(2, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

TEST(SGD, StepMovesAgainstTheGradient) {
    // loss = sum(w * c), so grad = c = [3, 4]; w - 0.1 * grad = [0.7, 1.6]
    Tensor w = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor c = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32);
    SGD optimizer({w}, 0.1f);

    F.sum(F.mul(w, c)).backward();
    optimizer.step();

    std::vector<float> values = to_vector<float>(w);
    EXPECT_FLOAT_EQ(values[0], 0.7f);
    EXPECT_FLOAT_EQ(values[1], 1.6f);
}

TEST(SGD, ZeroGradClearsGrads) {
    Tensor w = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    SGD optimizer({w}, 0.1f);

    F.sum(F.mul(w, w)).backward();
    ASSERT_NE(w.get_grad(), nullptr);

    optimizer.zero_grad();
    EXPECT_EQ(w.get_grad(), nullptr);
}

TEST(SGD, ParameterWithoutGradIsUnchanged) {
    Tensor w = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    SGD optimizer({w}, 0.1f);

    optimizer.step();

    EXPECT_EQ(to_vector<float>(w), (std::vector<float>{1, 2}));
}

TEST(SGD, StepUpdatesTheLayersOwnWeights) {
    Linear layer(2, 1);
    set_values(layer.parameters()[0], {1, 1});
    SGD optimizer(layer.parameters(), 0.5f);

    // loss = sum(x @ W + b) with x = [[1, 2]]: dW = [[1], [2]], db = [[1]]
    Tensor x = make_tensor<float>({1, 2}, {1, 2}, Dtype::FLOAT32);
    F.sum(layer.forward(x)).backward();
    optimizer.step();

    EXPECT_EQ(to_vector<float>(layer.parameters()[0]), (std::vector<float>{0.5f, 0}));
    EXPECT_EQ(to_vector<float>(layer.parameters()[1]), (std::vector<float>{-0.5f}));
}
