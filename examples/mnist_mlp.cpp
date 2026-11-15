// Trains a 784 -> 128 -> 10 network on raw MNIST pixels to recognize handwritten digits
//
// Usage: mnist_mlp [data folder] (default: data)
// The folder must hold the four unzipped MNSIT files : train-images-idx3-ubtye,
// train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "mllib/autograd.h"
#include "mllib/data.h"
#include "mllib/function.h"
#include "mllib/linear.h"
#include "mllib/optimizer.h"
#include "mllib/tensor.h"

// Fraction of rows where the hightest logit is the correct label
double accuracy(const Tensor& logits, const Tensor& labels) {
    int64_t n = logits.get_shape()[0];
    int64_t c = logits.get_shape()[1];
    const float* x = static_cast<const float*>(logits.get_data()->get_ptr());
    const int64_t* y = static_cast<const int64_t*>(labels.get_data()->get_ptr());

    int64_t correct = 0;
    for (int64_t i = 0; i < n; i++) {
        const float* row = x + i * c;
        int64_t predicted = std::max_element(row, row+c) - row;
        if (predicted == y[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / n;
}

void train(const std::string& dir) {
    Tensor train_images = load_images(dir + "/train-images.idx3-ubyte");
    Tensor train_labels = load_labels(dir + "/train-labels.idx1-ubyte");
    Tensor test_images = load_images(dir + "/t10k-images.idx3-ubyte");
    Tensor test_labels = load_labels(dir + "/t10k-labels.idx1-ubyte");
    std::printf("loaded %lld training and %lld test images\n",
                (long long)train_images.get_shape()[0], (long long)test_images.get_shape()[0]);
    


    Function F("train", {}, nullptr);
    Linear layer1(784, 128);
    Linear layer2(128,10);

    std::vector<Tensor> params = layer1.parameters();
    for (const Tensor& p : layer2.parameters()) {
        params.push_back(p);
    }
    SGD optmizer(params, 0.1f);

    const int epochs = 10;
    const int64_t batch_size = 64;
    int64_t n = train_images.get_shape()[0];

    // visit the training images in a new random order each epoch
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 generator(0);

    for (int epoch =1; epoch <= epochs; epoch++) {
        auto start_time = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), generator);

        int64_t step = 0;
        for (int64_t start = 0; start < n; start += batch_size, step++) {
            std::vector<int64_t> rows(order.begin() + start, order.begin() + std::min(start + batch_size, n));
            Tensor x = select_rows(train_images, rows);
            Tensor y = select_rows(train_labels, rows);

            Tensor logits = layer2.forward(F.relu(layer1.forward(x)));
            Tensor loss = F.cross_entropy(logits, y);

            optmizer.zero_grad();
            loss.backward();
            optmizer.step();

            if (step % 200 == 0) {
                std::printf("epoch %d step %4lld loss %.4f\n", epoch, (long long)step,
                            *static_cast<float*>(loss.get_data()->get_ptr()));
            }
        }

        // test accuracy, with grad tracking off
        Autograd::set_grad_enabled(false);
        Tensor test_logits = layer2.forward(F.relu(layer1.forward(test_images)));
        Autograd::set_grad_enabled(true);

        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        std::printf("epoch %d done in %.1fs test accuracy %.2f%%\n\n", epoch, seconds,
                100.0 * accuracy(test_logits, test_labels));
    }
    
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    try {
        train(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: mnist_mlp [folder holding the 4 unzipped MNIST files]");
        return 1;
    }
    return 0;
}
