// Trains a small convolutional netowrk on MNIST:
// (N, 1, 28, 28) -> conv 3x3 stride 2-> (N, 8, 14, 14) -> relu
//                -> cov 3x3 stride 2-> (N, 16, 7,7) -> rely
//                 -> flatten (N, 784) -> linear -> (N, 128) -> relu -> linear (N, 10)
// The stride-2 convolution halve the image size, so no pooling layer is needed

// Usage: mnist_cnn [data folder] (default: data)
// The folder must hold the four unzipped MNSIT files

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "mllib/autograd.h"
#include "mllib/conv2d.h"
#include "mllib/data.h"
#include "mllib/function.h"
#include "mllib/linear.h"
#include "mllib/optimizer.h"
#include "mllib/tensor.h"

struct Cnn {
    Conv2d conv1{1, 8, 3, 2, 1};
    Conv2d conv2{8, 16, 3, 2, 1};
    Linear fc1{16 * 7 * 7, 128};
    Linear fc2{128, 10};

    // images: (N, 784) rows straight from load images
    Tensor forward(Function& F, const Tensor& images) const {
        int64_t n = images.get_shape()[0];
        Tensor x = F.reshape(images, {n, 1, 28, 28});
        x = F.relu(conv1.forward(x));
        x = F.relu(conv2.forward(x));
        x = F.relu(fc1.forward(F.reshape(x, {n, -1})));
        return fc2.forward(x);
    }

    std::vector<Tensor> parameters() const {
        std::vector<Tensor> params = conv1.parameters();
        for (const Tensor& p: conv2.parameters()) {
            params.push_back(p);
        }
        for (const Tensor& p : fc1.parameters()) {
            params.push_back(p);
        }
        for (const Tensor& p : fc2.parameters()) {
            params.push_back(p);
        
        }
        return params;
    }
};

// Shifts pixels from 0..1 to mean 0, std 1, using MNIST's average (0.1307) and spread (0.3081).
// With only positive pixels, all the weights in a kernel get pushed the same way at once,
// which makes learning zig-zag and slow.
void normalize(const Tensor& images) {
    float* pixels = static_cast<float*>(images.get_data()->get_ptr());
    for (size_t i = 0; i < images.num_elements(); ++i) {
        pixels[i] = (pixels[i] - 0.1307f) / 0.3081f;
    }
}

// Fraction of rows where the highest logit is the correct label
double accuracy(const Tensor& logits, const Tensor& labels) {
    int64_t n = logits.get_shape()[0];
    int64_t c = logits.get_shape()[1];
    const float* x = static_cast<const float*>(logits.get_data()->get_ptr());
    const int64_t* y = static_cast<const int64_t*>(labels.get_data()->get_ptr());

    int64_t correct = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float* row = x + i * c;
        int64_t predicted = std::max_element(row, row + c) - row;
        if (predicted == y[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / n;
}

// Test accuracy over the whole test set, 1000 images at a time to keep memory small
double test_accuracy(Function& F, const Cnn& model, const Tensor& images, const Tensor& labels) {
    Autograd::set_grad_enabled(false);   // no backward() follows, so don't build the graph
    int64_t n = images.get_shape()[0];
    double correct = 0;
    for (int64_t start = 0; start < n; start += 1000) {
        std::vector<int64_t> rows(static_cast<size_t>(std::min<int64_t>(1000, n - start)));
        std::iota(rows.begin(), rows.end(), start);
        Tensor logits = model.forward(F, select_rows(images, rows));
        correct += accuracy(logits, select_rows(labels, rows)) * rows.size();
    }
    Autograd::set_grad_enabled(true);
    return correct / n;
}

void train(const std::string& dir) {
    Tensor train_images = load_images(dir + "/train-images.idx3-ubyte");
    Tensor train_labels = load_labels(dir + "/train-labels.idx1-ubyte");
    Tensor test_images = load_images(dir + "/t10k-images.idx3-ubyte");
    Tensor test_labels = load_labels(dir + "/t10k-labels.idx1-ubyte");
    normalize(train_images);
    normalize(test_images);
    std::printf("loaded %lld training and %lld test images\n",
                (long long)train_images.get_shape()[0], (long long)test_images.get_shape()[0]);

    Function F("train", {}, nullptr);
    Cnn model;

    const int epochs = 5;
    const int64_t batch_size = 64;
    int64_t n = train_images.get_shape()[0];

    // visit the training images in a new random order each epoch
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 generator(0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto start_time = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), generator);

        // big steps first, then smaller ones to settle in: 0.1, 0.05, 0.025, ...
        // (plain SGD keeps no state between steps, so a new one per epoch is fine)
        float lr = 0.1f / static_cast<float>(1 << (epoch - 1));
        SGD optimizer(model.parameters(), lr);

        int64_t step = 0;
        for (int64_t start = 0; start < n; start += batch_size, ++step) {
            std::vector<int64_t> rows(order.begin() + start, order.begin() + std::min(start + batch_size, n));
            Tensor x = select_rows(train_images, rows);
            Tensor y = select_rows(train_labels, rows);

            Tensor loss = F.cross_entropy(model.forward(F, x), y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if (step % 200 == 0) {
                std::printf("epoch %d  step %4lld  loss %.4f\n", epoch, (long long)step,
                            *static_cast<float*>(loss.get_data()->get_ptr()));
            }
        }

        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        std::printf("epoch %d done in %.1fs  test accuracy %.2f%%\n\n", epoch, seconds,
                    100.0 * test_accuracy(F, model, test_images, test_labels));
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    try {
        train(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: mnist_cnn [folder holding the 4 unzipped MNIST files] (default: data)\n");
        return 1;
    }
    return 0;
}

