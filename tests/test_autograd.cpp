// Unit tests for transpose, reshape's -1 dimension, and backward: each op's gradient and the
// autograd engine that runs them.
// The Transpose/Reshape/Backward/Autograd tests check values worked out by hand; the GradCheck
// tests compare each op's backward against a numerical estimate.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mllib/autograd.h"
#include "mllib/function.h"
#include "mllib/storage.h"
#include "mllib/tensor.h"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// The ops are non-static members, so the tests call them through F
Function F{"test", {}, nullptr};

// a[b][i][j] = 100b + 10i + j, shape (2, 2, 3)
const std::vector<float> kCube = {0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112};

// ---------------------------------------------------------------------------
// Gradient check: autograd's gradient vs. (loss(x + h) - loss(x - h)) / 2h
// ---------------------------------------------------------------------------

Tensor random_tensor(std::vector<int64_t> shape, double lo, double hi, unsigned seed,
                     bool requires_grad = true) {
    size_t n = 1;
    for (int64_t dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(lo, hi);
    std::vector<double> values(n);
    for (double& v : values) {
        v = distribution(generator);
    }
    return make_tensor(values, std::move(shape), Dtype::FLOAT64, requires_grad);
}

using Op = std::function<Tensor(const std::vector<Tensor>&)>;

// loss = sum(op(inputs) * w), with fixed random weights w. With a plain sum every
// incoming gradient would be 1, and a backward that ignored grad_output would still pass.
double loss_value(const Op& op, const std::vector<Tensor>& inputs, const Tensor& w) {
    return to_vector<double>(F.sum(F.mul(op(inputs), w)))[0];
}

// Inputs must be FLOAT64 so the numerical estimate is accurate
void expect_gradients_match(const Op& op, const std::vector<Tensor>& inputs) {
    Tensor w = random_tensor(op(inputs).get_shape(), -1.0, 1.0, 1234, false);
    F.sum(F.mul(op(inputs), w)).backward();

    const double h = 1e-6;
    for (size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE("input " + std::to_string(i));
        ASSERT_NE(inputs[i].get_grad(), nullptr);
        std::vector<double> autograd = to_vector<double>(*inputs[i].get_grad());
        double* x = static_cast<double*>(inputs[i].get_data()->get_ptr());

        for (size_t j = 0; j < inputs[i].num_elements(); ++j) {
            double original = x[j];
            x[j] = original + h;
            double loss_plus = loss_value(op, inputs, w);
            x[j] = original - h;
            double loss_minus = loss_value(op, inputs, w);
            x[j] = original;

            double numeric = (loss_plus - loss_minus) / (2 * h);
            EXPECT_NEAR(autograd[j], numeric, 1e-6 * std::max(1.0, std::abs(numeric)))
                << "element " << j;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// reshape, transpose (forward)
// ---------------------------------------------------------------------------

TEST(Reshape, InfersMinusOneDimension) {
    Tensor a = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::FLOAT32);
    Tensor out = F.reshape(a, {3, -1});

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(Transpose, TwoDimensionalValuesAndShape) {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    Tensor a = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(a, 0, 1);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST(Transpose, LastTwoDimsWithNegativeIndices) {
    // out[b][j][i] = a[b][i][j]: each 2x3 matrix is transposed, the batch dim is untouched
    Tensor a = make_tensor(kCube, {2, 2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(a, -2, -1);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 3, 2}));
    EXPECT_EQ(to_vector<float>(out),
              (std::vector<float>{0, 10, 1, 11, 2, 12, 100, 110, 101, 111, 102, 112}));
}

TEST(Transpose, OuterDimsOfThreeDimensionalTensor) {
    // out[j][i][b] = a[b][i][j]
    Tensor a = make_tensor(kCube, {2, 2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(a, 0, 2);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{3, 2, 2}));
    EXPECT_EQ(to_vector<float>(out),
              (std::vector<float>{0, 100, 10, 110, 1, 101, 11, 111, 2, 102, 12, 112}));
}

TEST(Transpose, TransposingTwiceGivesTheOriginal) {
    Tensor a = make_tensor(kCube, {2, 2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(F.transpose(a, 0, 2), 0, 2);

    EXPECT_EQ(out.get_shape(), a.get_shape());
    EXPECT_EQ(to_vector<float>(out), kCube);
}

TEST(Transpose, SameDimTwiceIsACopy) {
    Tensor a = make_tensor(kCube, {2, 2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(a, 1, 1);

    EXPECT_EQ(out.get_shape(), a.get_shape());
    EXPECT_EQ(to_vector<float>(out), kCube);
}

TEST(Transpose, OutputDoesNotShareStorage) {
    Tensor a = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::FLOAT32);
    Tensor out = F.transpose(a, 0, 1);

    EXPECT_NE(out.get_data(), a.get_data());
}

TEST(Transpose, EveryDtypeIsCopiedByteForByte) {
    Tensor i8  = make_tensor<int8_t>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::INT8);
    Tensor i64 = make_tensor<int64_t>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::INT64);
    Tensor f64 = make_tensor<double>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::FLOAT64);

    EXPECT_EQ(to_vector<int8_t>(F.transpose(i8, 0, 1)), (std::vector<int8_t>{1, 4, 2, 5, 3, 6}));
    EXPECT_EQ(to_vector<int64_t>(F.transpose(i64, 0, 1)), (std::vector<int64_t>{1, 4, 2, 5, 3, 6}));
    EXPECT_EQ(to_vector<double>(F.transpose(f64, 0, 1)), (std::vector<double>{1, 4, 2, 5, 3, 6}));
}

TEST(Transpose, DimOutOfRangeThrows) {
    Tensor a = make_tensor(kCube, {2, 2, 3}, Dtype::FLOAT32);
    Tensor scalar = make_tensor<float>({5}, {}, Dtype::FLOAT32);

    EXPECT_THROW(F.transpose(a, 0, 3), std::runtime_error);
    EXPECT_THROW(F.transpose(a, -4, 0), std::runtime_error);
    EXPECT_THROW(F.transpose(scalar, 0, 0), std::runtime_error);
}

TEST(Transpose, RequiresGradPropagates) {
    Tensor tracked   = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, true);
    Tensor untracked = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, false);

    EXPECT_TRUE(F.transpose(tracked, 0, 1).get_requires_grad());
    EXPECT_FALSE(F.transpose(untracked, 0, 1).get_requires_grad());
}

// ---------------------------------------------------------------------------
// Backward of each op, worked out by hand
// ---------------------------------------------------------------------------

TEST(Backward, AddPassesGradientToBothInputs) {
    Tensor a = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({5, 6, 7, 8}, {2, 2}, Dtype::FLOAT32, true);
    F.sum(F.add(a, b)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{1, 1, 1, 1}));
    EXPECT_EQ(grad_of(b), (std::vector<float>{1, 1, 1, 1}));
}

TEST(Backward, SubNegatesTheSecondInput) {
    Tensor a = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32, true);
    F.sum(F.sub(a, b)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{1, 1}));
    EXPECT_EQ(grad_of(b), (std::vector<float>{-1, -1}));
}

TEST(Backward, MulGivesEachInputTheOther) {
    Tensor a = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32, true);
    F.sum(F.mul(a, b)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{3, 4}));
    EXPECT_EQ(grad_of(b), (std::vector<float>{1, 2}));
}

TEST(Backward, Div) {
    // d(a/b)/da = 1/b = [0.5, 0.25],  d(a/b)/db = -a/b^2 = [-6/4, -8/16]
    Tensor a = make_tensor<float>({6, 8}, {2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({2, 4}, {2}, Dtype::FLOAT32, true);
    F.sum(F.div(a, b)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{0.5f, 0.25f}));
    EXPECT_EQ(grad_of(b), (std::vector<float>{-1.5f, -0.5f}));
}

TEST(Backward, Matmul) {
    // loss = sum(A @ B): dA = ones @ B^T (row sums of B), dB = A^T @ ones (row sums of A^T)
    Tensor a = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({5, 6, 7, 8}, {2, 2}, Dtype::FLOAT32, true);
    F.sum(F.matmul(a, b)).backward();

    EXPECT_EQ(grad_of(a), (std::vector<float>{11, 15, 11, 15}));
    EXPECT_EQ(grad_of(b), (std::vector<float>{4, 4, 6, 6}));
}

TEST(Backward, MeanSpreadsGradientEvenly) {
    Tensor x = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, true);
    F.mean(x).backward();

    EXPECT_EQ(grad_of(x), (std::vector<float>{0.25f, 0.25f, 0.25f, 0.25f}));
}

TEST(Backward, ReshapeAndTransposeRestoreTheInputShape) {
    Tensor x = make_tensor<float>({1, 2, 3, 4, 5, 6}, {6}, Dtype::FLOAT32, true);
    Tensor w = make_tensor<float>({10, 20, 30, 40, 50, 60}, {3, 2}, Dtype::FLOAT32);

    // x -> (2, 3) -> (3, 2); element (i, j) of the result is x[j * 3 + i]
    Tensor y = F.transpose(F.reshape(x, {2, -1}), 0, 1);
    F.sum(F.mul(y, w)).backward();

    ASSERT_EQ(x.get_grad()->get_shape(), (std::vector<int64_t>{6}));
    EXPECT_EQ(grad_of(x), (std::vector<float>{10, 30, 50, 20, 40, 60}));
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

TEST(Autograd, TensorUsedTwiceGetsBothGradients) {
    // loss = sum(x*x + x), so dloss/dx = 2x + 1
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    F.sum(F.add(F.mul(x, x), x)).backward();

    EXPECT_EQ(grad_of(x), (std::vector<float>{3, 5}));
}

TEST(Autograd, SharedIntermediateGetsEveryUseBeforePassingItOn) {
    // h = x*c feeds both p = h*d and q = h + e, so dloss/dh = d + 1 and dloss/dx = c * (d + 1)
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor c = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32);
    Tensor d = make_tensor<float>({5, 6}, {2}, Dtype::FLOAT32);
    Tensor e = make_tensor<float>({7, 8}, {2}, Dtype::FLOAT32);

    Tensor h = F.mul(x, c);
    F.sum(F.add(F.mul(h, d), F.add(h, e))).backward();

    EXPECT_EQ(grad_of(x), (std::vector<float>{18, 28}));
}

TEST(Autograd, CopiesOfATensorShareItsGrad) {
    // What the optimizer relies on: parameters() returns copies, and they must see the grad
    Tensor w = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor c = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32);
    std::vector<Tensor> params = {w};

    F.sum(F.mul(w, c)).backward();

    EXPECT_EQ(grad_of(params[0]), (std::vector<float>{3, 4}));
}

TEST(Autograd, SecondBackwardAddsToLeafGrads) {
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor c = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32);
    Tensor loss = F.sum(F.mul(x, c));

    loss.backward();
    loss.backward();

    EXPECT_EQ(grad_of(x), (std::vector<float>{6, 8}));
}

TEST(Autograd, InputThatDoesNotRequireGradGetsNoGrad) {
    Tensor a = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    Tensor b = make_tensor<float>({3, 4}, {2}, Dtype::FLOAT32, false);
    F.sum(F.mul(a, b)).backward();

    EXPECT_EQ(b.get_grad(), nullptr);
}

TEST(Autograd, NoGraphIsBuiltWhileGradIsDisabled) {
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);

    Autograd::set_grad_enabled(false);
    Tensor y = F.add(x, x);
    Autograd::set_grad_enabled(true);

    EXPECT_EQ(y.get_grad_fn(), nullptr);
    EXPECT_NE(F.add(x, x).get_grad_fn(), nullptr);
}

TEST(Autograd, BackwardLeavesGradTrackingOn) {
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    F.sum(F.mul(x, x)).backward();

    EXPECT_TRUE(Autograd::is_grad_enabled());
}

TEST(Autograd, BackwardOnNonScalarThrows) {
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, true);
    EXPECT_THROW(F.add(x, x).backward(), std::runtime_error);
}

TEST(Autograd, BackwardWithoutRequiresGradThrows) {
    Tensor x = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32, false);
    EXPECT_THROW(F.sum(x).backward(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Gradient checks (FLOAT64)
// ---------------------------------------------------------------------------

TEST(GradCheck, Add) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.add(in[0], in[1]); },
                           {random_tensor({2, 3}, -2, 2, 1), random_tensor({2, 3}, -2, 2, 2)});
}

TEST(GradCheck, Sub) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.sub(in[0], in[1]); },
                           {random_tensor({2, 3}, -2, 2, 3), random_tensor({2, 3}, -2, 2, 4)});
}

TEST(GradCheck, Mul) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.mul(in[0], in[1]); },
                           {random_tensor({2, 3}, -2, 2, 5), random_tensor({2, 3}, -2, 2, 6)});
}

TEST(GradCheck, Div) {
    // denominator kept away from zero
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.div(in[0], in[1]); },
                           {random_tensor({2, 3}, -2, 2, 7), random_tensor({2, 3}, 0.5, 2, 8)});
}

TEST(GradCheck, Matmul) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.matmul(in[0], in[1]); },
                           {random_tensor({2, 3}, -2, 2, 9), random_tensor({3, 4}, -2, 2, 10)});
}

TEST(GradCheck, Sum) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.sum(in[0]); },
                           {random_tensor({2, 3}, -2, 2, 11)});
}

TEST(GradCheck, Mean) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.mean(in[0]); },
                           {random_tensor({2, 3}, -2, 2, 12)});
}

TEST(GradCheck, Reshape) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.reshape(in[0], {3, -1}); },
                           {random_tensor({2, 3}, -2, 2, 13)});
}

TEST(GradCheck, Transpose) {
    expect_gradients_match([](const std::vector<Tensor>& in) { return F.transpose(in[0], -2, -1); },
                           {random_tensor({2, 3, 4}, -2, 2, 14)});
}

TEST(GradCheck, ChainWithReusedIntermediate) {
    expect_gradients_match(
        [](const std::vector<Tensor>& in) {
            Tensor xt = F.transpose(F.reshape(in[0], {3, -1}), 0, 1); // (6) -> (3, 2) -> (2, 3)
            Tensor h = F.matmul(xt, in[1]);                           // (2, 2)
            return F.mul(h, F.sub(h, in[2]));                         // h used twice
        },
        {random_tensor({6}, -2, 2, 15), random_tensor({3, 2}, -2, 2, 16),
         random_tensor({2, 2}, -2, 2, 17)});
}
