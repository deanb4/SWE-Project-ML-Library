// Unit tests for the tensor operations in Function: add, sub, mul, div, matmul, sum, mean.
// Every expected value below was worked out by hand, never by calling the code under test.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "mllib/function.h"
#include "mllib/storage.h"
#include "mllib/tensor.h"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a CPU tensor holding `values` with the given shape.
template<typename T>
Tensor make_tensor(const std::vector<T>& values, std::vector<int64_t> shape,
                   Dtype dtype, bool requires_grad = false) {
    size_t n = 1;
    for (int64_t dim : shape) {
        n *= static_cast<size_t>(dim);
    }

    // Fail loudly on a typo in a test instead of reading garbage memory
    if (values.size() != n) {
        throw std::invalid_argument("make_tensor: number of values does not match shape");
    }
    if (sizeof(T) != dtype_size(dtype)) {
        throw std::invalid_argument("make_tensor: C++ type does not match dtype");
    }

    auto storage = std::make_shared<Storage>(n * sizeof(T), Device::CPU);
    if (n > 0) {
        std::memcpy(storage->get_ptr(), values.data(), n * sizeof(T));
    }
    return Tensor(storage, std::move(shape), Device::CPU, requires_grad, dtype);
}

// Copy a tensor's elements into a vector so a whole result can be compared at once.
template<typename T>
std::vector<T> to_vector(const Tensor& t) {
    const T* ptr = static_cast<const T*>(t.get_data()->get_ptr());
    return std::vector<T>(ptr, ptr + t.num_elements());
}

// Run `fn` and return the message of the std::runtime_error it throws ("" if it doesn't throw).
template<typename Fn>
std::string error_message(Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

// The ops are non-static members, so every test needs a Function instance to call them on.
class FunctionTest : public ::testing::Test {
    protected:
        Function F{"test", nullptr, nullptr};
};

// Aliases so the output reads "Add.Float32Values" rather than "FunctionTest.AddFloat32Values"
using Add         = FunctionTest;
using Sub         = FunctionTest;
using Mul         = FunctionTest;
using Div         = FunctionTest;
using Elementwise = FunctionTest;
using Matmul      = FunctionTest;
using Sum         = FunctionTest;
using Mean        = FunctionTest;

// Inputs for the element-wise ops, chosen so every result is exact in floating point
const std::vector<float> kA = {6, 8, 10, 12};
const std::vector<float> kB = {2, 4, 5, 3};

// Per-dtype checks: each one runs the same small example through one case of the op's switch
template<typename T>
void expect_add_values(Function& F, Dtype dtype, const char* name) {
    SCOPED_TRACE(name);
    Tensor a = make_tensor<T>({6, 8, 10, 12}, {2, 2}, dtype);
    Tensor b = make_tensor<T>({2, 4, 5, 3}, {2, 2}, dtype);
    Tensor out = F.add(a, b);
    EXPECT_EQ(out.get_dtype(), dtype);
    EXPECT_EQ(to_vector<T>(out), (std::vector<T>{8, 12, 15, 15}));
}

template<typename T>
void expect_matmul_values(Function& F, Dtype dtype, const char* name) {
    SCOPED_TRACE(name);
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]], small enough to fit in INT8
    Tensor a = make_tensor<T>({1, 2, 3, 4}, {2, 2}, dtype);
    Tensor b = make_tensor<T>({5, 6, 7, 8}, {2, 2}, dtype);
    Tensor out = F.matmul(a, b);
    EXPECT_EQ(out.get_dtype(), dtype);
    EXPECT_EQ(to_vector<T>(out), (std::vector<T>{19, 22, 43, 50}));
}

template<typename T>
void expect_sum_values(Function& F, Dtype dtype, const char* name) {
    SCOPED_TRACE(name);
    Tensor t = make_tensor<T>({1, 2, 3, 4}, {2, 2}, dtype);
    Tensor out = F.sum(t);
    EXPECT_EQ(out.get_dtype(), dtype);
    EXPECT_EQ(to_vector<T>(out), (std::vector<T>{10}));
}

}  // namespace

// ---------------------------------------------------------------------------
// add, sub, mul, div
// ---------------------------------------------------------------------------

TEST_F(Add, Float32Values) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    Tensor out = F.add(a, b);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(out.get_dtype(), Dtype::FLOAT32);
    EXPECT_EQ(out.get_device(), Device::CPU);
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{8, 12, 15, 15}));
}

TEST_F(Sub, Float32Values) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.sub(a, b)), (std::vector<float>{4, 4, 5, 9}));
}

TEST_F(Sub, ArgumentOrderMatters) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.sub(b, a)), (std::vector<float>{-4, -4, -5, -9}));
}

TEST_F(Mul, Float32Values) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.mul(a, b)), (std::vector<float>{12, 32, 50, 36}));
}

TEST_F(Div, Float32Values) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.div(a, b)), (std::vector<float>{3, 2, 2, 4}));
}

TEST_F(Div, ArgumentOrderMatters) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    std::vector<float> out = to_vector<float>(F.div(b, a));

    ASSERT_EQ(out.size(), 4u);
    EXPECT_FLOAT_EQ(out[0], 2.0f / 6.0f);
    EXPECT_FLOAT_EQ(out[1], 0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 0.25f);
}

TEST_F(Div, IntegerDivisionTruncatesTowardZero) {
    Tensor a = make_tensor<int32_t>({7, -7}, {2}, Dtype::INT32);
    Tensor b = make_tensor<int32_t>({2, 2}, {2}, Dtype::INT32);
    EXPECT_EQ(to_vector<int32_t>(F.div(a, b)), (std::vector<int32_t>{3, -3}));
}

TEST_F(Div, FloatDivisionByZeroGivesInfAndNaN) {
    Tensor a = make_tensor<float>({1, -1, 0}, {3}, Dtype::FLOAT32);
    Tensor b = make_tensor<float>({0, 0, 0}, {3}, Dtype::FLOAT32);
    std::vector<float> out = to_vector<float>(F.div(a, b));

    ASSERT_EQ(out.size(), 3u);
    EXPECT_TRUE(std::isinf(out[0]) && out[0] > 0);
    EXPECT_TRUE(std::isinf(out[1]) && out[1] < 0);
    EXPECT_TRUE(std::isnan(out[2]));
}

// Integer division by zero is deliberately not tested: it is a hardware fault, not an
// exception, and would crash the whole test run. Decide how div should handle it first.

TEST_F(Add, EveryDtype) {
    expect_add_values<double>(F, Dtype::FLOAT64, "FLOAT64");
    expect_add_values<float>(F, Dtype::FLOAT32, "FLOAT32");
    expect_add_values<int64_t>(F, Dtype::INT64, "INT64");
    expect_add_values<int32_t>(F, Dtype::INT32, "INT32");
    expect_add_values<int16_t>(F, Dtype::INT16, "INT16");
    expect_add_values<int8_t>(F, Dtype::INT8, "INT8");
    expect_add_values<uint8_t>(F, Dtype::UINT8, "UINT8");
}

TEST_F(Add, ThreeDimensionalShapeIsPreserved) {
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 1.0f);  // 1, 2, ..., 24
    Tensor t = make_tensor(values, {2, 3, 4}, Dtype::FLOAT32);
    Tensor out = F.add(t, t);

    std::vector<float> expected(24);
    for (size_t i = 0; i < 24; ++i) {
        expected[i] = 2.0f * static_cast<float>(i + 1);
    }
    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(to_vector<float>(out), expected);
}

TEST_F(Add, ZeroDimensionalTensor) {
    Tensor a = make_tensor<float>({5}, {}, Dtype::FLOAT32);
    Tensor b = make_tensor<float>({3}, {}, Dtype::FLOAT32);
    Tensor out = F.add(a, b);

    EXPECT_TRUE(out.get_shape().empty());
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{8}));
}

TEST_F(Add, RequiresGradPropagates) {
    Tensor tracked   = make_tensor(kA, {2, 2}, Dtype::FLOAT32, true);
    Tensor untracked = make_tensor(kB, {2, 2}, Dtype::FLOAT32, false);

    EXPECT_TRUE(F.add(tracked, untracked).get_requires_grad());
    EXPECT_TRUE(F.add(untracked, tracked).get_requires_grad());
    EXPECT_FALSE(F.add(untracked, untracked).get_requires_grad());
}

TEST_F(Add, InputsAreNotModified) {
    Tensor a = make_tensor(kA, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor(kB, {2, 2}, Dtype::FLOAT32);
    F.add(a, b);

    EXPECT_EQ(to_vector<float>(a), kA);
    EXPECT_EQ(to_vector<float>(b), kB);
}

TEST_F(Elementwise, ShapeMismatchErrorNamesTheOp) {
    Tensor a = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor<float>({1, 2, 3, 4}, {4}, Dtype::FLOAT32);

    EXPECT_EQ(error_message([&] { F.add(a, b); }), "add: shape mismatch");
    EXPECT_EQ(error_message([&] { F.sub(a, b); }), "sub: shape mismatch");
    EXPECT_EQ(error_message([&] { F.mul(a, b); }), "mul: shape mismatch");
    EXPECT_EQ(error_message([&] { F.div(a, b); }), "div: shape mismatch");
}

TEST_F(Elementwise, DtypeMismatchErrorNamesTheOp) {
    Tensor a = make_tensor<float>({1, 2}, {2}, Dtype::FLOAT32);
    Tensor b = make_tensor<double>({1, 2}, {2}, Dtype::FLOAT64);

    EXPECT_EQ(error_message([&] { F.add(a, b); }), "add: dtype mismatch");
    EXPECT_EQ(error_message([&] { F.sub(a, b); }), "sub: dtype mismatch");
    EXPECT_EQ(error_message([&] { F.mul(a, b); }), "mul: dtype mismatch");
    EXPECT_EQ(error_message([&] { F.div(a, b); }), "div: dtype mismatch");
}

TEST_F(Elementwise, BoolTensorsThrow) {
    Tensor t = make_tensor<uint8_t>({1, 0}, {2}, Dtype::BOOL);

    EXPECT_THROW(F.add(t, t), std::runtime_error);
    EXPECT_THROW(F.sub(t, t), std::runtime_error);
    EXPECT_THROW(F.mul(t, t), std::runtime_error);
    EXPECT_THROW(F.div(t, t), std::runtime_error);
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

TEST_F(Matmul, NonSquareValuesAndShape) {
    // Non-square on purpose: a square test would still pass if M and N were swapped
    Tensor a = make_tensor<float>({1, 2, 3,
                                   4, 5, 6}, {2, 3}, Dtype::FLOAT32);
    Tensor b = make_tensor<float>({1,  2,  3,  4,
                                   5,  6,  7,  8,
                                   9, 10, 11, 12}, {3, 4}, Dtype::FLOAT32);
    Tensor out = F.matmul(a, b);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 4}));
    EXPECT_EQ(out.get_dtype(), Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{38, 44, 50, 56,
                                                          83, 98, 113, 128}));
}

TEST_F(Matmul, IdentityLeavesMatrixUnchanged) {
    std::vector<float> values = {1, 2, 3,
                                 4, 5, 6};
    Tensor a = make_tensor(values, {2, 3}, Dtype::FLOAT32);
    Tensor identity = make_tensor<float>({1, 0, 0,
                                          0, 1, 0,
                                          0, 0, 1}, {3, 3}, Dtype::FLOAT32);
    Tensor out = F.matmul(a, identity);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(to_vector<float>(out), values);
}

TEST_F(Matmul, OneByOne) {
    Tensor a = make_tensor<float>({3}, {1, 1}, Dtype::FLOAT32);
    Tensor b = make_tensor<float>({4}, {1, 1}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.matmul(a, b)), (std::vector<float>{12}));
}

TEST_F(Matmul, EveryDtype) {
    expect_matmul_values<double>(F, Dtype::FLOAT64, "FLOAT64");
    expect_matmul_values<float>(F, Dtype::FLOAT32, "FLOAT32");
    expect_matmul_values<int64_t>(F, Dtype::INT64, "INT64");
    expect_matmul_values<int32_t>(F, Dtype::INT32, "INT32");
    expect_matmul_values<int16_t>(F, Dtype::INT16, "INT16");
    expect_matmul_values<int8_t>(F, Dtype::INT8, "INT8");
    expect_matmul_values<uint8_t>(F, Dtype::UINT8, "UINT8");
}

TEST_F(Matmul, InnerDimensionMismatchThrows) {
    Tensor a = make_tensor<float>({1, 2, 3, 4, 5, 6}, {2, 3}, Dtype::FLOAT32);
    EXPECT_THROW(F.matmul(a, a), std::runtime_error);  // (2x3) @ (2x3)
}

TEST_F(Matmul, NonMatrixInputThrows) {
    Tensor vec    = make_tensor<float>({1, 2, 3}, {3}, Dtype::FLOAT32);
    Tensor column = make_tensor<float>({1, 2, 3}, {3, 1}, Dtype::FLOAT32);
    Tensor cube   = make_tensor<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}, Dtype::FLOAT32);

    EXPECT_THROW(F.matmul(vec, column), std::runtime_error);
    EXPECT_THROW(F.matmul(cube, cube), std::runtime_error);
}

TEST_F(Matmul, DtypeMismatchThrows) {
    Tensor a = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32);
    Tensor b = make_tensor<double>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT64);
    EXPECT_THROW(F.matmul(a, b), std::runtime_error);
}

TEST_F(Matmul, BoolTensorsThrow) {
    Tensor t = make_tensor<uint8_t>({1, 0, 0, 1}, {2, 2}, Dtype::BOOL);
    EXPECT_THROW(F.matmul(t, t), std::runtime_error);
}

TEST_F(Matmul, RequiresGradPropagates) {
    Tensor tracked   = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, true);
    Tensor untracked = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32, false);

    EXPECT_TRUE(F.matmul(tracked, untracked).get_requires_grad());
    EXPECT_TRUE(F.matmul(untracked, tracked).get_requires_grad());
    EXPECT_FALSE(F.matmul(untracked, untracked).get_requires_grad());
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

TEST_F(Sum, Float32Values) {
    Tensor t = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32);
    Tensor out = F.sum(t);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{1}));
    EXPECT_EQ(out.get_dtype(), Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{10}));
}

TEST_F(Sum, SingleElement) {
    Tensor t = make_tensor<float>({5}, {1}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.sum(t)), (std::vector<float>{5}));
}

TEST_F(Sum, NegativeValues) {
    Tensor t = make_tensor<float>({-1, -2, 3}, {3}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.sum(t)), (std::vector<float>{0}));
}

TEST_F(Sum, ThreeDimensionalSumsEverything) {
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 1.0f);  // 1 + 2 + ... + 24 = 300
    Tensor t = make_tensor(values, {2, 3, 4}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.sum(t)), (std::vector<float>{300}));
}

TEST_F(Sum, EveryDtype) {
    expect_sum_values<double>(F, Dtype::FLOAT64, "FLOAT64");
    expect_sum_values<float>(F, Dtype::FLOAT32, "FLOAT32");
    expect_sum_values<int64_t>(F, Dtype::INT64, "INT64");
    expect_sum_values<int32_t>(F, Dtype::INT32, "INT32");
    expect_sum_values<int16_t>(F, Dtype::INT16, "INT16");
    expect_sum_values<int8_t>(F, Dtype::INT8, "INT8");
    expect_sum_values<uint8_t>(F, Dtype::UINT8, "UINT8");
}

TEST_F(Sum, EmptyTensorThrows) {
    Tensor t = make_tensor<float>({}, {0}, Dtype::FLOAT32);
    EXPECT_THROW(F.sum(t), std::runtime_error);
}

TEST_F(Sum, BoolTensorThrows) {
    Tensor t = make_tensor<uint8_t>({1, 0}, {2}, Dtype::BOOL);
    EXPECT_THROW(F.sum(t), std::runtime_error);
}

TEST_F(Sum, RequiresGradPropagates) {
    EXPECT_TRUE(F.sum(make_tensor(kA, {2, 2}, Dtype::FLOAT32, true)).get_requires_grad());
    EXPECT_FALSE(F.sum(make_tensor(kA, {2, 2}, Dtype::FLOAT32, false)).get_requires_grad());
}

TEST_F(Sum, Float32LargeInputStaysAccurate) {
    // 10 million copies of 0.1f should add up to about 1,000,000. A float running total
    // drifts far from that (adding 0.1 to a large float loses digits); a double total doesn't.
    const size_t n = 10'000'000;
    Tensor t = make_tensor(std::vector<float>(n, 0.1f), {static_cast<int64_t>(n)}, Dtype::FLOAT32);
    EXPECT_NEAR(to_vector<float>(F.sum(t))[0], 1'000'000.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

TEST_F(Mean, Float32Values) {
    // 2.5 rather than a whole number, so integer-style truncation would be caught
    Tensor t = make_tensor<float>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT32);
    Tensor out = F.mean(t);

    EXPECT_EQ(out.get_shape(), (std::vector<int64_t>{1}));
    EXPECT_EQ(out.get_dtype(), Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(out), (std::vector<float>{2.5f}));
}

TEST_F(Mean, Float64Values) {
    Tensor t = make_tensor<double>({1, 2, 3, 4}, {2, 2}, Dtype::FLOAT64);
    Tensor out = F.mean(t);

    EXPECT_EQ(out.get_dtype(), Dtype::FLOAT64);
    EXPECT_EQ(to_vector<double>(out), (std::vector<double>{2.5}));
}

TEST_F(Mean, NegativeValues) {
    Tensor t = make_tensor<float>({-3, -1}, {2}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.mean(t)), (std::vector<float>{-2}));
}

TEST_F(Mean, ThreeDimensionalAveragesEverything) {
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 1.0f);  // 300 / 24 = 12.5
    Tensor t = make_tensor(values, {2, 3, 4}, Dtype::FLOAT32);
    EXPECT_EQ(to_vector<float>(F.mean(t)), (std::vector<float>{12.5f}));
}

TEST_F(Mean, IntegerDtypesThrow) {
    EXPECT_THROW(F.mean(make_tensor<int64_t>({1, 2}, {2}, Dtype::INT64)), std::runtime_error);
    EXPECT_THROW(F.mean(make_tensor<int32_t>({1, 2}, {2}, Dtype::INT32)), std::runtime_error);
    EXPECT_THROW(F.mean(make_tensor<int16_t>({1, 2}, {2}, Dtype::INT16)), std::runtime_error);
    EXPECT_THROW(F.mean(make_tensor<int8_t>({1, 2}, {2}, Dtype::INT8)), std::runtime_error);
    EXPECT_THROW(F.mean(make_tensor<uint8_t>({1, 2}, {2}, Dtype::UINT8)), std::runtime_error);
}

TEST_F(Mean, EmptyTensorThrows) {
    Tensor t = make_tensor<float>({}, {0}, Dtype::FLOAT32);
    EXPECT_THROW(F.mean(t), std::runtime_error);
}

TEST_F(Mean, BoolTensorThrows) {
    Tensor t = make_tensor<uint8_t>({1, 0}, {2}, Dtype::BOOL);
    EXPECT_THROW(F.mean(t), std::runtime_error);
}

TEST_F(Mean, RequiresGradPropagates) {
    EXPECT_TRUE(F.mean(make_tensor(kA, {2, 2}, Dtype::FLOAT32, true)).get_requires_grad());
    EXPECT_FALSE(F.mean(make_tensor(kA, {2, 2}, Dtype::FLOAT32, false)).get_requires_grad());
}

TEST_F(Mean, Float32LargeInputStaysAccurate) {
    // Same drift as the sum test: with a float running total this comes out near 0.1088
    const size_t n = 10'000'000;
    Tensor t = make_tensor(std::vector<float>(n, 0.1f), {static_cast<int64_t>(n)}, Dtype::FLOAT32);
    EXPECT_NEAR(to_vector<float>(F.mean(t))[0], 0.1f, 1e-6f);
}
